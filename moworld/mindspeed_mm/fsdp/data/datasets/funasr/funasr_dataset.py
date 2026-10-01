import logging
import random
from dataclasses import asdict

import torch

from funasr.register import tables

from mindspeed_mm.fsdp.utils.register import data_register

logger = logging.getLogger(__name__)


@tables.register("prompt_classes", "MultiContextPrompt")
class MultiContextPrompt:
    """
    Patched MultiContextPrompt class.
    Add your modifications here to override the original FunASR implementation.
    """
    CONTEXT_TEMPLATES = {
        'en': {
            'header': "Please combine the context information provided below to complete the speech transcription task more accurately. If there is no relevant information, we will leave it blank.\n",
            'fields': {
                'hist_context': "Historical transcription: {hist_context}\n",
                'one_pass_result': "One-pass result: {one_pass_result}\n",
                'hotwords': "Hotword list: {hotwords}\n"
            }
        },
        'zh': {
            'header': "请结合下面提供的上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n",
            'fields': {
                'hist_context': "历史转写结果：{hist_context}\n",
                'one_pass_result': "一遍解码结果：{one_pass_result}\n",
                'hotwords': "热词列表：{hotwords}\n"
            }
        }
    }

    def __init__(self,
                 use_hist=True,
                 use_one_pass_result=True,
                 use_hotwords=True,
                 use_asr_hotwords=True,
                 use_multi_lingual_prompt=True,
                 **kwargs):
        self.use_hist = use_hist
        self.use_one_pass_result = use_one_pass_result
        self.use_hotwords = use_hotwords
        self.use_asr_hotwords = use_asr_hotwords
        self.use_multi_lingual_prompt = use_multi_lingual_prompt
        self.kwargs = kwargs

        chinese_hotwords_list = kwargs.get("chinese_hotwords_list", "")
        english_hotwords_list = kwargs.get("english_hotwords_list", "")
        if chinese_hotwords_list:
            self.chinese_hotwords_list, self.chinese_hotwords_num = self.get_hotwords_list(chinese_hotwords_list)
        else:
            self.chinese_hotwords_list = None
            self.chinese_hotwords_num = 0
        logging.info(f"chinese_hotwords_num: {self.chinese_hotwords_num}")

        if english_hotwords_list:
            self.english_hotwords_list, self.english_hotwords_num = self.get_hotwords_list(english_hotwords_list)
        else:
            self.english_hotwords_list = None
            self.english_hotwords_num = 0
        logging.info(f"english_hotwords_num: {self.english_hotwords_num}")

        self.max_neg_hotwords_num = kwargs.get("max_neg_hotwords_num", 900)
        self.min_neg_hotwords_num = kwargs.get("min_neg_hotwords_num", 0)

    def get_hotwords_list(self, hotwords_file):
        with open(hotwords_file, "r") as f:
            hotwords_list = f.read().strip().split("\n")
        return hotwords_list, len(hotwords_list)

    def detect_language(self, text):
        if isinstance(text, list):
            text = " ".join(text)

        chinese_count = 0
        english_count = 0
        total_count = 0

        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                chinese_count += 1
            elif 'A' <= char <= 'Z' or 'a' <= char <= 'z':
                english_count += 1
            total_count += 1

        if total_count == 0:
            return 'zh'

        if (chinese_count > english_count) and (chinese_count / total_count > 0.3):
            return 'zh'
        else:
            return 'en'

    def hotwords_sampling(self, hotwords):
        hotwords_list = hotwords
        selected_hotwords = []
        if self.max_neg_hotwords_num > -1:
            max_neg_hotwords_num = min(self.max_neg_hotwords_num, len(hotwords_list))
        else:
            max_neg_hotwords_num = len(hotwords_list)

        if self.min_neg_hotwords_num < max_neg_hotwords_num:
            selected_hotwords_num = random.randint(self.min_neg_hotwords_num, max_neg_hotwords_num)
        else:
            selected_hotwords_num = max_neg_hotwords_num
        if selected_hotwords_num > 0:
            selected_hotwords = random.sample(hotwords_list, selected_hotwords_num)

        return selected_hotwords, selected_hotwords_num

    def get_prompt(self, item, language):
        template = self.CONTEXT_TEMPLATES[language]

        prompt = template['header']

        context_lines = []

        if self.use_hist and item.get("hist_context"):
            context_lines.append(template['fields']['hist_context'].format(hist_context=item["hist_context"]))

        if self.use_one_pass_result and item.get("one_pass_result"):
            context_lines.append(template['fields']['one_pass_result'].format(one_pass_result=item["one_pass_result"]))

        hotwords = None
        if self.use_hotwords and item.get("hotwords"):
            hotwords = item["hotwords"]
        if self.use_asr_hotwords and item.get("asr_hotwords"):
            hotwords = item["asr_hotwords"]
        if hotwords is not None and hotwords != "":
            language = self.detect_language(hotwords)
            if language == 'en':
                neg_hotwords = self.english_hotwords_list
            else:
                neg_hotwords = self.chinese_hotwords_list
            if neg_hotwords is not None:
                selected_neg_hotwords, selected_neg_hotwords_num = self.hotwords_sampling(neg_hotwords)
            else:
                selected_neg_hotwords = []

            if not isinstance(hotwords, list):
                pos_hotwords = hotwords.split(", ")
            else:
                pos_hotwords = hotwords
            hotwords = pos_hotwords + selected_neg_hotwords
            random.shuffle(hotwords)
            hotwords = ", ".join(hotwords)
            context_lines.append(template['fields']['hotwords'].format(hotwords=hotwords))

        if context_lines:
            prompt += ''.join(context_lines)
        else:
            prompt += "\n\n\n"

        return prompt

    def get_inference_prompt(self, item, language="zh"):
        template = self.CONTEXT_TEMPLATES[language]

        prompt = template['header']

        context_lines = []

        if self.use_hist and item.get("hist_context"):
            context_lines.append(template['fields']['hist_context'].format(hist_context=item["hist_context"]))

        if self.use_one_pass_result and item.get("one_pass_result"):
            context_lines.append(template['fields']['one_pass_result'].format(one_pass_result=item["one_pass_result"]))

        hotwords = None
        if self.use_hotwords and item.get("hotwords"):
            hotwords = item["hotwords"]
        if self.use_asr_hotwords and item.get("asr_hotwords"):
            hotwords = item["asr_hotwords"]
        if hotwords is not None and hotwords != "":
            logging.info(f"hotwords: {hotwords}")
            language = self.detect_language(hotwords)
            if language == 'en':
                neg_hotwords = self.english_hotwords_list
            else:
                neg_hotwords = self.chinese_hotwords_list
            if neg_hotwords is not None:
                selected_neg_hotwords, selected_neg_hotwords_num = self.hotwords_sampling(neg_hotwords)
            else:
                selected_neg_hotwords = []

            if not isinstance(hotwords, list):
                pos_hotwords = hotwords.split(", ")
            else:
                pos_hotwords = hotwords
            hotwords = pos_hotwords + selected_neg_hotwords
            logging.info(f"selected_neg_hotwords_num: {selected_neg_hotwords_num}")
            random.shuffle(hotwords)
            hotwords = ", ".join(hotwords)
            context_lines.append(template['fields']['hotwords'].format(hotwords=hotwords))

        if context_lines:
            prompt += ''.join(context_lines)
        else:
            prompt += "\n\n\n"

        return prompt


@tables.register("prompt_classes", "MultiContextPromptNew")
class MultiContextPromptNew:
    CONTEXT_TEMPLATES = {
        'en': {
            'header': "Please combine the context information to complete the speech transcription task more accurately. If there is no relevant information, we will leave it blank.\n\n",
            'context_header': "**Context:**\n",
            'fields': {
                'hist_context': "Historical transcription: {hist_context}\n",
                'one_pass_result': "One-pass result: {one_pass_result}\n",
                'hotwords': "Hotword list: {hotwords}\n"
            }
        },
        'zh': {
            'header': "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n",
            'context_header': "**上下文信息：**\n",
            'fields': {
                'hist_context': "历史转写结果：{hist_context}\n",
                'one_pass_result': "一遍解码结果：{one_pass_result}\n",
                'hotwords': "热词列表：{hotwords}\n"
            }
        }
    }

    def __init__(self,
                 use_hist=True,
                 use_one_pass_result=True,
                 use_hotwords=True,
                 use_multi_lingual_prompt=True,
                 **kwargs):
        self.use_hist = use_hist
        self.use_one_pass_result = use_one_pass_result
        self.use_hotwords = use_hotwords
        self.use_multi_lingual_prompt = use_multi_lingual_prompt

        self.use_full_hotwords_ratio = kwargs.get("use_full_hotwords_ratio", 0.2)
        self.max_hotwords_num = kwargs.get("max_hotwords_num", -1)
        self.min_hotwords_num = kwargs.get("min_hotwords_num", 15)

    def hotwords_sampling(self, hotwords):

        hotwords_list = hotwords.split(", ")
        if self.max_hotwords_num > 0:
            max_hotwords_num = min(self.max_hotwords_num, len(hotwords_list))
        else:
            max_hotwords_num = len(hotwords_list)

        if self.min_hotwords_num < max_hotwords_num:
            selected_hotwords_num = random.randint(self.min_hotwords_num, max_hotwords_num)
        else:
            selected_hotwords_num = max_hotwords_num

        selected_hotwords = random.sample(hotwords_list, selected_hotwords_num)
        hotwords_list = ", ".join(selected_hotwords)

        return hotwords_list, selected_hotwords_num

    def get_prompt(self, item, language):
        template = self.CONTEXT_TEMPLATES[language]

        prompt = template['header']

        context_lines = []

        if self.use_hist and item.get("hist_context"):
            context_lines.append(template['fields']['hist_context'].format(hist_context=item["hist_context"]))

        if self.use_one_pass_result and item.get("one_pass_result"):
            context_lines.append(template['fields']['one_pass_result'].format(one_pass_result=item["one_pass_result"]))

        if self.use_hotwords and item.get("hotwords"):
            hotwords = item["hotwords"]
            if random.random() < self.use_full_hotwords_ratio:
                hotwords = hotwords
            else:
                hotwords, selected_hotwords_num = self.hotwords_sampling(hotwords)
            context_lines.append(template['fields']['hotwords'].format(hotwords=hotwords))

        if context_lines:
            prompt += template['context_header'] + ''.join(context_lines)

        return prompt

    def get_inference_prompt(self, hist_context="", one_pass_result="", hotwords=""):
        language = 'zh' if self.use_multi_lingual_prompt and random.random() < 0.5 else 'en'
        template = self.CONTEXT_TEMPLATES[language]

        prompt = template['header']

        context_lines = []

        if hist_context:
            context_lines.append(template['fields']['hist_context'].format(hist_context=hist_context))
        if one_pass_result:
            context_lines.append(template['fields']['one_pass_result'].format(one_pass_result=one_pass_result))
        if hotwords:
            context_lines.append(template['fields']['hotwords'].format(hotwords=hotwords))

        if context_lines:
            prompt += template['context_header'] + ''.join(context_lines)

        return prompt


@tables.register("dataloader_classes", "DataloaderMapStyle")
class DataloaderMapStyle:
    def __init__(self, frontend=None, tokenizer=None, **kwargs):
        # dataset
        logging.info("Build dataloader")

        dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
        dataset_tr = None
        # split dataset
        self.data_split_num = kwargs["dataset_conf"].get("data_split_num", 1)
        if self.data_split_num == 1:
            dataset_tr = dataset_class(
                kwargs.get("train_data_set_list"),
                frontend=frontend,
                tokenizer=tokenizer,
                is_training=True,
                **kwargs.get("dataset_conf"),
            )
        dataset_val = dataset_class(
            kwargs.get("valid_data_set_list"),
            frontend=frontend,
            tokenizer=tokenizer,
            is_training=False,
            **kwargs.get("dataset_conf"),
        )

        self.dataset_tr = dataset_tr
        self.dataset_val = dataset_val
        self.kwargs = kwargs

        self.dataset_class = dataset_class
        self.frontend = frontend
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def build_iter(self, epoch=0, data_split_i=0, start_step=0, **kwargs):
        # Check if we have a cached dataloader that we can reuse
        cache_key = f"cache_{data_split_i}_{start_step}"

        # For data_split_num = 1 and start_step = 0, we can reuse cached dataloaders
        if self.data_split_num == 1 and start_step == 0:
            if hasattr(self, '_cached_dataloaders') and cache_key in self._cached_dataloaders:
                # Get cached dataloaders
                dataloader_tr, dataloader_val = self._cached_dataloaders[cache_key]
                # Update epoch on batch samplers for proper shuffling
                if hasattr(dataloader_tr.batch_sampler, 'set_epoch'):
                    dataloader_tr.batch_sampler.set_epoch(epoch)
                if hasattr(dataloader_val.batch_sampler, 'set_epoch'):
                    dataloader_val.batch_sampler.set_epoch(epoch)
                return dataloader_tr, dataloader_val

        # reload dataset slice
        if self.data_split_num > 1:
            del self.dataset_tr
            self.dataset_tr = self.dataset_class(
                self.kwargs.get("train_data_set_list"),
                frontend=self.frontend,
                tokenizer=self.tokenizer,
                is_training=True,
                **self.kwargs.get("dataset_conf"),
                data_split_i=data_split_i,
            )

        # dataloader
        batch_sampler = self.kwargs["dataset_conf"].get("batch_sampler", "BatchSampler")
        batch_sampler_val = None
        if batch_sampler is not None:
            batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
            batch_sampler = batch_sampler_class(
                self.dataset_tr, start_step=start_step, **self.kwargs.get("dataset_conf")
            )
            batch_sampler_val = batch_sampler_class(
                self.dataset_val, is_training=False, **self.kwargs.get("dataset_conf")
            )

        # Set epoch on batch samplers for proper shuffling
        batch_sampler["batch_sampler"].set_epoch(epoch)
        batch_sampler_val["batch_sampler"].set_epoch(epoch)

        # Add persistent_workers to dataloader arguments
        if "num_workers" in batch_sampler and batch_sampler["num_workers"] > 0:
            batch_sampler.setdefault("persistent_workers", True)
            batch_sampler_val.setdefault("persistent_workers", True)

        # Create dataloader instances
        dataloader_tr = torch.utils.data.DataLoader(
            self.dataset_tr, collate_fn=self.dataset_tr.collator, **batch_sampler
        )
        dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val, collate_fn=self.dataset_val.collator, **batch_sampler_val
        )

        # Cache dataloaders when data_split_num = 1 and start_step = 0
        if self.data_split_num == 1 and start_step == 0:
            if not hasattr(self, '_cached_dataloaders'):
                self._cached_dataloaders = {}
            self._cached_dataloaders[cache_key] = (dataloader_tr, dataloader_val)

        return dataloader_tr, dataloader_val


def build_funasr_dataloader_factory(data_args, frontend, tokenizer):
    """Build FunASR dataloader factory with proper distributed config."""
    dataset_conf = data_args.dataset_param.dataset_conf
    dataloader_conf = data_args.dataset_param.dataset_conf
    combined_conf = {**dataloader_conf, **dataset_conf}

    dataloader_kwargs = {
        'dataset': "FunASR",
        'train_data_set_list': data_args.train_data_set_list,
        'valid_data_set_list': data_args.valid_data_set_list,
        'dataset_conf': combined_conf,
        'frontend': frontend,
        'tokenizer': tokenizer,
    }

    dl_class = tables.dataloader_classes.get(
        dataset_conf.get('dataloader', 'DataloaderMapStyle')
    )
    return dl_class(**dataloader_kwargs)

data_register.register("funasr")(build_funasr_dataloader_factory)
