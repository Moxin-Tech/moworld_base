# Copyright 2025 the LlamaFactory team.
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import Dict, List, Any
from collections import defaultdict
from mindspeed_mm.data.data_utils.func_utils.log import get_logger
logger = get_logger(__file__)


def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    from mindspore._c_expression import disable_multi_thread
    disable_multi_thread()
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        try:
            tool_schema = []
            if '_tools' in examples:
                tool_schema = examples['_tools'][i]

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                tools=tool_schema
            )
        except OSError as e:
            err_img = examples["_images"][i] if examples["_images"][i] else "No images"
            logger.warning(f"Skipping invalid sample: {err_img}. Error: {str(e)}")
            continue

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])
        model_inputs["audios"].append(examples["_audios"][i])

    return model_inputs
