# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch


def get_cache_policy(layer_number, cache_policy_init, cache_interval):
    cache_policy = cache_policy_init
    if cache_interval != 0:
        if layer_number % (cache_interval + 1) == 1:
            cache_policy = cache_policy_init
        else:
            cache_policy = None

    return cache_policy


class ContextParallelKVCache:
    """Context Parallelism KV Cache Implementation"""

    def __init__(self, cache_policy, outer_data, inner_data, k, v) -> None:
        self.outer_size, self.outer_ring_p2p = outer_data
        self.inner_size, self.inner_ring_p2p = inner_data
        self.cache_policy = cache_policy
        self.k = k
        self.v = v
        self.cp_size = self.outer_size * self.inner_size
        self.outer_index = 0

        enable_mla = k[-1].shape[-1] != v[-1].shape[-1]
        if not enable_mla:
            send_data = torch.zeros((2, *self.k[-1].shape), dtype=self.k[-1].dtype, device=self.k[-1].device)
            send_data.copy_(torch.cat((self.k[-1].unsqueeze(0), self.v[-1].unsqueeze(0)), dim=0))
            outer_recv_data = send_data.clone()
            inner_recv_data = send_data.clone()
        else:
            send_data = [self.k[-1].clone(), self.v[-1].clone()]
            outer_recv_data = [self.k[-1].clone(), self.v[-1].clone()]
            inner_recv_data = [self.k[-1].clone(), self.v[-1].clone()]
        self.cur_kv, self.outer_next_kv, self.inner_next_kv = send_data, outer_recv_data, inner_recv_data

        self.k_out, self.v_out = None, None

    def communicate_outer_ring_kv(self, index, shapes=None) -> None:
        """
        Implements of kv communications in outer ring

        Args:
            index (int): the index of outer for loop
        """
        self.outer_index = index

        # index > 0, using kv after communication
        if index > 0:
            if index == 1 and self.cache_policy == "half":
                # special case: index=1, cache_policy=half, KV block should be transformed to K
                self.outer_ring_p2p.wait()
                if self.inner_size > 1:
                    # KV have been transformed in inner ring
                    self.cur_kv.copy_(self.outer_next_kv[1])
                    self.outer_next_kv = self.outer_next_kv[1].clone()
                else:
                    # KV is not transformed in inner ring
                    self.cur_kv, self.outer_next_kv = self.outer_next_kv, self.cur_kv
                    self.k_out, self.v_out = self.cur_kv[0].clone(), self.cur_kv[1].clone()
                    self.cur_kv = self.cur_kv[1].clone()
                    self.outer_next_kv = self.outer_next_kv[1].clone()
            else:
                self.outer_ring_p2p.wait()
                self.cur_kv, self.outer_next_kv = self.outer_next_kv, self.cur_kv

        # last step, no need to communicate KV
        is_last_step = index + 1 == self.outer_size

        # only need communicate KV in the first step when full cache
        first_step_with_full_cache = self.cache_policy == "full" and index > 0

        if not first_step_with_full_cache and not is_last_step:
            if isinstance(self.cur_kv, (list, tuple)):
                send_tensor = [x.clone() for x in self.cur_kv]
            else:
                send_tensor = self.cur_kv.clone()
            self.outer_ring_p2p.async_send_recv(send_tensor, self.outer_next_kv, shapes=shapes)

    def communicate_inner_ring_kv(self, index, shapes=None):
        """
        Implements of kv communications in inner ring

        Args:
            index (int): the index of inner for loop

        Returns:
            cur_k (torch.tensor): k(keys), backward operator input in this iteration
            cur_v (torch.tensor): v(values), backward operator input in this iteration
        """
        total_index = self.outer_index * self.inner_size + index

        # index > 0, using kv after communication
        if index > 0:
            if total_index == 1 and self.cache_policy == "half":
                # special case: index=1, cache_policy=half, KV block should be transformed to K
                self.inner_ring_p2p.wait()
                self.cur_kv, self.inner_next_kv = self.inner_next_kv, self.cur_kv
                self.k_out, self.v_out = self.cur_kv[0].clone(), self.cur_kv[1].clone()
                self.cur_kv = self.cur_kv[1].clone()
                self.inner_next_kv = self.inner_next_kv[1].clone()
            else:
                self.inner_ring_p2p.wait()
                self.cur_kv, self.inner_next_kv = self.inner_next_kv, self.cur_kv

        # last step, no need to communicate KV
        is_last_step = index + 1 == self.inner_size

        # only need communicate KV in the first step when full cache
        first_step_with_full_cache = self.cache_policy == "full" and total_index > 0

        if not first_step_with_full_cache and not is_last_step:
            self.inner_ring_p2p.async_send_recv(self.cur_kv, self.inner_next_kv, shapes=shapes)

        cache_index = self.cp_size - total_index - 1
        if self.cache_policy is None:
            self.k_out, self.v_out = self.cur_kv[0], self.cur_kv[1]

        elif self.cache_policy == "half":
            if total_index == 0:
                self.k_out, self.v_out = self.cur_kv[0], self.cur_kv[1]
            elif total_index > 1:
                self.k_out, self.v_out = self.k[cache_index], self.cur_kv

        elif self.cache_policy == "full":
            if total_index <= 1:
                self.k_out, self.v_out = self.cur_kv[0], self.cur_kv[1]
            else:
                self.k_out, self.v_out = self.k[cache_index], self.v[cache_index]

        return self.k_out, self.v_out
