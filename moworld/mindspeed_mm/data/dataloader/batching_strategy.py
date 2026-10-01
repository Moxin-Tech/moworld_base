# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class DynBszBuffer:
    """
    A buffer to store samples for dynamic batch size.
    """

    def __init__(self):
        self._buffer = []
        self._buffer_sample_lens = []
        self.del_idxs = []
        self.cur_idx = 0
        self.all_token_cnt = 0

    def append(self, item: dict):
        """
        Append a sample to the buffer.
        Args:
            item: a sample to append to the buffer.
                The sample should be a dict with the following keys:
                    - input_ids: torch.Tensor of shape (seq_len, )
                    - attention_mask: torch.Tensor of shape (seq_len, )
        """
        self._buffer.append(item)
        self._buffer_sample_lens.append(item["attention_mask"].sum())
        self.all_token_cnt += self._buffer_sample_lens[-1]

    def get_samples(self, max_seq_len: int, force: bool = True):
        """
        get samples from the buffer.
        Args:
            max_seq_len: the number of tokens to get.
            force: if True, the first sample will be returned even if it is not full.
        Returns:
            samples: a list of samples.
        """
        cum_seq_len = 0
        samples = []
        while self.cur_idx < len(self._buffer) and cum_seq_len < max_seq_len:
            seq_len = self._buffer_sample_lens[self.cur_idx]
            is_valid_sample = self.cur_idx not in self.del_idxs # if current sample is not used
            """
            In these cases, a sample could add to sequence:
                1. force is true and current sample is the first sample in this sequence (see function annotation)
                2. current sequence length + cumulate length < max sequence length
            """
            could_add_to_seq = (force is True and cum_seq_len == 0) or (seq_len <= max_seq_len - cum_seq_len)
            if is_valid_sample and could_add_to_seq:
                cum_seq_len += seq_len
                samples.append(self._buffer[self.cur_idx])
                self.del_idxs.append(self.cur_idx)
            self.cur_idx += 1
        if len(samples) == 0:
            raise ValueError("Could not get samples from buffer")
        return samples

    def __len__(self):
        return len(self._buffer)

    def flush(self):
        """
        Flush the buffer.
        """
        self.cur_idx = 0
        self.all_token_cnt -= sum([self._buffer_sample_lens[idx] for idx in self.del_idxs])
        buffer_len = len(self._buffer)
        self._buffer = [self._buffer[idx] for idx in range(buffer_len) if idx not in self.del_idxs]
        self._buffer_sample_lens = [
            self._buffer_sample_lens[idx]
            for idx in range(buffer_len)
            if idx not in self.del_idxs
        ]
        self.del_idxs = []

    def merge(self, buffer_to_merge: "DynBszBuffer"):
        """ "
        Merge the buffer with another buffer.
        Args:
            buffer_to_merge: the buffer to merge.
        """
        self.flush()
        buffer_to_merge.flush()
        for item in buffer_to_merge._buffer:
            self.append(item)


class TextBatchingStrategy(object):
    """ "
    Batching strategy for text data.
    Args:
        max_seq_len: the max number of tokens to get for each sequence.
        buffer_size: the size of the buffer.
    """

    def __init__(
        self,
        max_seq_len,
        buffer_size: int = 500,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size  # minimum samples in buffer
        self._buffer = DynBszBuffer()

    def is_full_filled(self) -> bool:
        return len(self._buffer) >= self.buffer_size and self._buffer.all_token_cnt >= self.max_seq_len

    def put_item(self, item: dict):
        self._buffer.append(item)

    def get_micro_batch(self) -> list:
        samples = self._buffer.get_samples(self.max_seq_len)
        self._buffer.flush()  # remove the selected samples.
        return samples

    def empty(self) -> bool:
        return len(self._buffer) == 0
