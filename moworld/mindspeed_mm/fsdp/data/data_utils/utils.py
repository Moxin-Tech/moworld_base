import random
import itertools
import numpy as np
import torch


def build_iterations(train_dl=None, val_dl=None, test_dl=None, iterator_type="cyclic"):

    def _cyclic_iter(dl):
        while True:
            for x in dl:
                yield x

    def _get_iterator(dataloader, iter_type=iterator_type):
        """Return dataset iterator."""
        if iter_type == "single":
            return iter(dataloader)
        elif iter_type == "cyclic":
            return iter(_cyclic_iter(dataloader))
        else:
            raise NotImplementedError("unexpected iterator type")

    if train_dl is not None:
        train_data_iterator = _get_iterator(train_dl)
    else:
        train_data_iterator = None

    if val_dl is not None:
        valid_data_iterator = _get_iterator(val_dl)
    else:
        valid_data_iterator = None

    if test_dl is not None:
        test_data_iterator = _get_iterator(test_dl)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def get_seed_worker(seed):
    """Deterministic dataloader"""

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker
