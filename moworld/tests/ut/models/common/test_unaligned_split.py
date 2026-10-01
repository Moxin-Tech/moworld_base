from mindspeed_mm.models.common.communications import cal_split_sizes
from tests.ut.utils import judge_expression


class TestUnalignedSplit:
    def test_cal_split_sizes(self):
        dim_size = 100
        world_size = 3
        res = cal_split_sizes(dim_size, world_size)
        gt = [34, 33, 33]
        judge_expression(isinstance(res, list))
        judge_expression(res == gt)
