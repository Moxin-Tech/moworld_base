from abc import ABC, abstractmethod
import torch


class TPPattern:
    @staticmethod
    @abstractmethod
    def split(weight, tp_size):
        pass

    @staticmethod
    @abstractmethod
    def merge(weights):
        pass


class ColumnParallelTP(TPPattern):
    @staticmethod
    def split(weight, tp_size):
        return torch.chunk(weight, tp_size, dim=0)

    @staticmethod
    def merge(weights):
        return torch.cat(weights, dim=0)


class RowParallelTP(TPPattern):
    @staticmethod
    def split(weight, tp_size):
        return torch.chunk(weight, tp_size, dim=1)

    @staticmethod
    def merge(weights):
        return torch.cat(weights, dim=1)


class QKVfusedColumnTP(TPPattern):
    @staticmethod
    def split(weight, tp_size):
        wq, wk, wv = torch.chunk(weight, 3, dim=0)
        wqs = torch.chunk(wq, tp_size, dim=0)
        wks = torch.chunk(wk, tp_size, dim=0)
        wvs = torch.chunk(wv, tp_size, dim=0)
        weights = [torch.cat([wqs[i], wks[i], wvs[i]], dim=0) for i in range(tp_size)]
        return weights

    @staticmethod
    def merge(weights):
        chunked_weights = [torch.chunk(weight, 3, dim=0) for weight in weights]

        wqs = [chunk[0] for chunk in chunked_weights]
        wks = [chunk[1] for chunk in chunked_weights]
        wvs = [chunk[2] for chunk in chunked_weights]

        weight = torch.cat([
            torch.cat(wqs, dim=0),
            torch.cat(wks, dim=0),
            torch.cat(wvs, dim=0)
        ], dim=0)
        return weight


# tp pattern mapping
TP_PARTTERN_MAPPING = {
    "column_parallel_tp": ColumnParallelTP,
    "row_parallel_tp": RowParallelTP,
    "qkv_fused_column_tp": QKVfusedColumnTP
}
