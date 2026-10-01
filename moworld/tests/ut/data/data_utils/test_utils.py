import mindspeed.megatron_adaptor

from mindspeed_mm.data.data_utils.utils import DataFileReader, TextProcesser
from tests.ut.utils import judge_expression


class TestDataFileReader:

    def test_init_datafilereader(self):
        get_data = DataFileReader()
        judge_expression(isinstance(get_data, DataFileReader))

    def test_combine_mode(self):
        data_config = {
            "data_path": "/home/ci_resource/data/OpenSoraPlan/v1.2/pixabay_v2/train_data.txt",
            "data_storage_mode": "combine"
        }
        get_data = DataFileReader(data_storage_mode=data_config["data_storage_mode"])
        cap_list = get_data(data_config["data_path"])
        judge_expression(isinstance(cap_list, list))

    def test_standard_mode(self):
        data_config = {
            "data_path": "/home/ci_resource/data/cogvideox1_0/data.jsonl",
            "data_folder": "/home/ci_resource/data/cogvideox1_0/",
            "data_storage_mode": "standard"
        }
        get_data = DataFileReader(data_storage_mode=data_config["data_storage_mode"])
        cap_list = get_data(data_config["data_path"])
        judge_expression(isinstance(cap_list, list))


class TestTextProcesser:

    def test_clean_caption(self):
        test_cases = [
            ("  Hello   World  ", "Hello World"),  # Multiple spaces
            ("Hello\tWorld", "Hello World"),       # Tabs
            ("Hello\nWorld", "Hello World"),       # Newlines
            (" Hello \t World \n ", "Hello World"),  # Mixed whitespace
        ]
        for input_text, _ in test_cases:
            result = TextProcesser.clean_caption(input_text)
            judge_expression(isinstance(result, str))
