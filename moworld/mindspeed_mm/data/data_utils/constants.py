# some constants used for dataset
PROMPT_MASK = "prompt_mask"
PROMPT_IDS = "prompt_ids"
PROMPT_MASK_2 = "prompt_mask_2"
PROMPT_IDS_2 = "prompt_ids_2"
TEXT = "text"
VIDEO = "video"
VIDEO_REJECTED = "video_rejected"
PROMPT = "prompt"
LATENTS = "latents"
VIDEO_MASK = "video_mask"
MASKED_VIDEO = "masked_video"
INPUT_MASK = "input_mask"
FILE_INFO = "file"
FILE_REJECTED_INFO = "file_rejected"
CAPTIONS = "captions"
CAMERA_CONTROL_DIRECTION = "camera_control_direction"
CAM_FROM_WORLDS = "cam_from_worlds"
INTRINSICS = "intrinsics"
ORI_H = "ori_h"
ORI_W = "ori_w"
START_FRAME = "start_frame"
END_FRAME = "end_frame"
SCORE = "score"
SCORE_REJECTED = "score_rejected"
IMG_FPS = 120
SORA_MODEL_PROTECTED_KEYS = [
    PROMPT_MASK,
    PROMPT_IDS,
    PROMPT,
    TEXT,
    VIDEO,
    VIDEO_MASK,
    LATENTS,
]
MODEL_CONSTANTS = {
    'internvl': {
        "IMG_CONTEXT_TOKEN": "<IMG_CONTEXT>",
        "IGNORE_INDEX": -100,
        "IMG_START_TOKEN": "<img>",
        "IMG_END_TOKEN": "</img>"
    },
    'internvl2_5': {
        'IMG_CONTEXT_TOKEN': '<IMG_CONTEXT>',
        'IMG_START_TOKEN': '<img>',
        'IMG_END_TOKEN': '</img>',
        'QUAD_START_TOKEN': '<quad>',
        'QUAD_END_TOKEN': '</quad>',
        'REF_START_TOKEN': '<ref>',
        'REF_END_TOKEN': '</ref>',
        'BOX_START_TOKEN': '<box>',
        'BOX_END_TOKEN': '</box>',
        "IGNORE_INDEX": -100
    },
    "deepseekvl2": {
        "IGNORE_INDEX": -100,
        "IMAGE_TOKEN": "<image>"
    },
    "internvit_qwen3": {
        'IMG_CONTEXT_TOKEN': '<|image_pad|>',
        'IMG_START_TOKEN': '<|vision_start|>',
        'IMG_END_TOKEN': '<|vision_end|>',
    }
}

# some constants used for dataloader
# PrefetchGradAccDataLoader
GLOBAL_STEP_TOKEN_NUM = "global_step_token_num"
AVG_PER_STEP_TOKEN_NUM = "avg_per_step_token_num"
