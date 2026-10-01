import os
import json
import stat
import argparse

from transformers import AutoTokenizer
from ai_asset_obfuscate import ModelAssetObfuscation, ModelType, DataAssetObfuscation
from ai_asset_obfuscate.vision_api import ImageDataAssetObfuscation

from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer
from mindspeed_mm.data.data_utils.func_utils.mm_plugin import get_mm_plugin
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.convert import load_tokenizer


VOCAB_SIZE = 151936
IMAGE_MAX_PIXELS = 1048576  # 1024 * 1024
TOKEN_WHITE_LIST = [151643, 151644, 151645, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653,
                    151654, 151655, 151656, 151657, 151658, 151659, 151660, 151661, 151662, 151663, 151664,
                    151665, 151666, 151667, 151668, 198, 2610, 525, 264, 10950, 17847, 13, 872, 77091, 8948]


class Qwen3VLPMCC:
    def __init__(self, hf_model_path, obf_seed, model_type_enum=ModelType.QWEN3_VL):
        """
        hf_model_path: the huggingface model path
        obf_seed: obfuscate factor
        model_type_enum: model type for obfuscate
        """
        self.hf_model_path = hf_model_path
        self.obf_seed = obf_seed
        self.model_type_enum = model_type_enum

    def obfuscate_model(self, model_save_path="./obf_model", device_id=None):
        obf_model = ModelAssetObfuscation.create_model_obfuscation(
            self.hf_model_path, self.model_type_enum, token_white_list=TOKEN_WHITE_LIST, is_obfuscation=True
        )
        res = obf_model.set_seed_content(2, self.obf_seed)
        print(res)
        res = obf_model.model_weight_obf(2, model_save_path=model_save_path, device_type="npu", device_id=device_id)
        print(res)

    def obfuscate_data(
        self,
        src_json_path, src_img_dir,
        obf_json_path, obf_img_dir,
        data_limit=1000):
        """
        src_json_path: the origin mllm_format_json_path.
        src_img_dir: the origin image dir
        obf_json_path: the path for save obfuscated json
        obf_img_dir: the dir for save obfuscated image
        data_limit: the num of limited obfuscate data
        """
        if os.path.exists(obf_json_path):
            raise FileExistsError(f"{obf_json_path} already exists, please rename it or remove it")

        os.makedirs(obf_img_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path)
        template = get_template_and_fix_tokenizer(tokenizer, "qwen3_vl_nothink")
        mm_plugin = get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>")
        process_args = ProcessorArguments(self.hf_model_path)
        processor = load_tokenizer(process_args)['processor']
        setattr(processor, "image_max_pixels", IMAGE_MAX_PIXELS)

        image_obj = ImageDataAssetObfuscation()
        image_obj.set_seed_content(self.obf_seed)

        text_obj = DataAssetObfuscation(vocab_size=VOCAB_SIZE, token_white_list=TOKEN_WHITE_LIST)
        text_obj.set_seed_content(self.obf_seed)

        mllm_format_llava_instruct_data = []
        index = 0
        with open(src_json_path, "r") as f:
            info_json = json.loads(f.read())

        for index, item in enumerate(info_json):
            if index > data_limit:
                break
            if not item.get("image"):
                continue

            img_path = os.path.join(src_img_dir, item["image"])
            obf_img_path = os.path.join(obf_img_dir, item["image"])

            if not os.path.exists(img_path):
                print(f"{img_path} is not exists")
                continue

            print(f"Processing image: {img_path}, index: {index}")
            self._obf_image(img_path, obf_img_path, image_obj)

            new_item = {"images": [obf_img_path], "messages": []}
            tmp_item = []
            for turn in item["conversations"]:
                if turn["from"] == "human":
                    tmp_item.append({"role": "user", "content": turn["value"]})
                elif turn["from"] == "gpt":
                    tmp_item.append({"role": "assistant", "content": turn["value"]})
                else:
                    raise ValueError(f"unknown role: {turn['from']}")

            message = mm_plugin.process_messages(tmp_item, [obf_img_path], [], [], processor)
            tokens = template.encode_multiturn(tokenizer, message)
            for token in tokens:
                token_user = text_obj.data_1d_obf(token[0])
                token_assistant = text_obj.data_1d_obf(token[1])
                new_item["messages"].append({"role": "user", "content": token_user})
                new_item["messages"].append({"role": "assistant", "content": token_assistant})
            mllm_format_llava_instruct_data.append(new_item)

        output_json = json.dumps(mllm_format_llava_instruct_data, ensure_ascii=False)
        with os.fdopen(os.open(obf_json_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
            f.write(output_json)
        print(f"finish converting dataset into {obf_json_path}")

    def _obf_image(self, src_img_path, dst_img_path, image_obj):
        try:
            with open(src_img_path, 'rb') as f:
                data = bytearray(f.read())
            obf_data = image_obj.image_bytearray_obf(data)
            with open(dst_img_path, 'wb') as f:
                f.write(obf_data)
            print(f"Image obfuscation successful: {dst_img_path}")
        except Exception as e:
            print(f"Image obfuscation failed {dst_img_path}: {e}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obf-type",
                        choices=["model", "data"],
                        help="Choice to obfuscate model or data")
    parser.add_argument("--hf-model-path",
                        type=str, required=True,
                        help="Path to HF format checkpoint directory")
    parser.add_argument("--obf-seed",
                        type=str, required=True,
                        help="Obfuscate factor")

    parser = _add_obf_model_args(parser)
    parser = _add_obf_data_args(parser)
    return parser


def _add_obf_model_args(parser):
    group = parser.add_argument_group(title="obf model args")
    group.add_argument('--model-save-path',
                       type=str, default='./obf_model',
                       help='Path to save obfucated model ckpt')
    group.add_argument('--device-id',
                       nargs='+', type=int, default=[0],
                       help='Device id to parallel process ckpt')
    return parser


def _add_obf_data_args(parser):
    group = parser.add_argument_group(title="obf data args")
    group.add_argument('--src-json-path',
                       type=str,
                       help='Path to origin mllm_format_json_path')
    group.add_argument('--src-img-dir',
                       type=str,
                       help='Path to origin image folder')
    group.add_argument('--obf-json-path',
                       type=str, default='./obf_json.json',
                       help='Path to save obfuscated json')
    group.add_argument('--obf-img-dir',
                       type=str, default='./obf_images',
                       help='Path to save obfuscated image')
    group.add_argument('--data-limit',
                       type=int, default=1000,
                       help='Number of data obfuscate')
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    pmcc = Qwen3VLPMCC(args.hf_model_path, args.obf_seed)

    if args.obf_type == "model":
        pmcc.obfuscate_model(model_save_path=args.model_save_path, device_id=args.device_id)
    elif args.obf_type == "data":
        if args.src_json_path is None or args.src_img_dir is None:
            raise ValueError("Both --src-json-path and --src-img-dir must be provided.")
        pmcc.obfuscate_data(args.src_json_path, args.src_img_dir, args.obf_json_path, args.obf_img_dir, args.data_limit)
