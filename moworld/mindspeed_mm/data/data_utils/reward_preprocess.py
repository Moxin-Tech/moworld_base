# Copyright 2025 The KwaiVGI team. All rights reserved.

import copy

import torch

VIDEOSCORE_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the frames of a given video and see the text prompt for generating the video,
then give scores based on its {dimension_name}, i.e., {dimension_description}.
Output a float number from 1.0 to 5.0 for this dimension,
the higher the number is, the better the video performs in that sub-score,
the lowest 1.0 means Bad, the highest 5.0 means Perfect/Real (the video is like a real video).
The text prompt used for generation is "{text_prompt}".
"""

DIMENSION_DESCRIPTIONS = {
    'VQ': ['visual quality', 'the quality of the video in terms of clearness, resolution, brightness, and color'],
    'TA': ['text-to-video alignment', 'the alignment between the text prompt and the video content and motion'],
    'MQ': ['motion quality', 'the quality of the motion in terms of consistency, smoothness, and completeness'],
    'Overall': ['Overall Performance', 'the overall performance of the video in terms of visual quality, text-to-video alignment, and motion quality'],
}

SIMPLE_PROMPT = """
Please evaluate the {dimension_name} of a generated video. Consider {dimension_description}.
The text prompt used for generation is "{text_prompt}".
"""

DETAILED_PROMPT_WITH_SPECIAL_TOKEN = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

Please provide the ratings of Visual Quality: <|VQ_reward|>
END

**Motion Quality:**
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

Please provide the ratings of Motion Quality: <|MQ_reward|>
END

**Text Alignment:**
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Text Alignment: <|TA_reward|>
END
"""

DETAILED_PROMPT = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

**Motion Quality:**
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.


**Text Alignment:**
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Visual Quality, Motion Quality, and Text Alignment.
"""

SIMPLE_PROMPT_NO_PROMPT = """
Please evaluate the {dimension_name} of a generated video. Consider {dimension_description}.
"""


def build_prompt(prompt, dimension, template_type):
    if isinstance(dimension, list) and len(dimension) > 1:
        dimension_name = ", ".join([DIMENSION_DESCRIPTIONS[d][0] for d in dimension])
        dimension_name = f'overall performance({dimension_name})'
        dimension_description = "the overall performance of the video"
    else:
        if isinstance(dimension, list):
            dimension = dimension[0]
        dimension_name = DIMENSION_DESCRIPTIONS[dimension][0]
        dimension_description = DIMENSION_DESCRIPTIONS[dimension][1]

    if template_type == "none":
        return prompt
    elif template_type == "simple":
        return SIMPLE_PROMPT.format(dimension_name=dimension_name,
                                    dimension_description=dimension_description,
                                    text_prompt=prompt)
    elif template_type == "video_score":
        return VIDEOSCORE_QUERY_PROMPT.format(dimension_name=dimension_name,
                                              dimension_description=dimension_description,
                                              text_prompt=prompt)
    elif template_type == "detailed_special":
        return DETAILED_PROMPT_WITH_SPECIAL_TOKEN.format(text_prompt=prompt)
    elif template_type == "detailed":
        return DETAILED_PROMPT.format(text_prompt=prompt)
    else:
        raise ValueError("Invalid template type")


def fill_data_template(
        data_folder,
        relative_path,
        prompt,
        fps,
        max_pixels,
        num_frames,
        eval_dim,
        sample_nframe=None,
        sample_type="uniform",
        prompt_template_type="detailed_special",
):
    data_dict = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{data_folder}/{relative_path}",
                    "max_pixels": max_pixels,
                    "fps": fps if sample_nframe is None else None,
                    "nframes": min(sample_nframe, num_frames) if sample_nframe is not None else None,
                    "sample_type": sample_type
                },
                {
                    "type": "text",
                    "text": build_prompt(prompt, eval_dim, prompt_template_type),
                },
            ],
        }
    ]
    return data_dict


def clean_examples(example):
    """
    remove unnecessary keys from message(very very necessary)
    """
    clean_example = copy.deepcopy(example)
    for i in range(len(example[0])):
        for key in example[0]['content'][i].keys():
            if example[0]['content'][i][key] is None:
                clean_example[0]['content'][i].pop(key)
    return clean_example


def build_reward_data(example, data_folder, eval_dim, fps, video_max_pixels, sample_nframe=None, sample_type="uniform",
                      prompt_template_type="detailed_special", **extra_block_kwargs):
    A_data = fill_data_template(data_folder=data_folder,
                                relative_path=example["path_A"],
                                prompt=example["prompt"],
                                fps=fps,
                                max_pixels=video_max_pixels,
                                num_frames=example["num_frames_A"],
                                eval_dim=eval_dim,
                                sample_nframe=sample_nframe,
                                sample_type=sample_type,
                                prompt_template_type=prompt_template_type)
    B_data = fill_data_template(data_folder=data_folder,
                                relative_path=example["path_B"],
                                prompt=example["prompt"],
                                fps=fps,
                                max_pixels=video_max_pixels,
                                num_frames=example["num_frames_B"],
                                eval_dim=eval_dim,
                                sample_nframe=sample_nframe,
                                sample_type=sample_type,
                                prompt_template_type=prompt_template_type)

    chosen_labels = []
    A_scores = []
    B_scores = []

    for eval_dim_ in eval_dim:
        # chosen_label: 1 if A is chosen, -1 if B is chosen, 0 if tied.
        # 22 if invalid.
        try:
            if example[f"{eval_dim_}"] is not None:
                if example[f"{eval_dim_}"] == "A":
                    chosen_label = 1
                elif example[f"{eval_dim_}"] == "B":
                    chosen_label = -1
                elif example[f"{eval_dim_}"] == "same":
                    chosen_label = 0
                elif example[f"{eval_dim_}"] == "invalid":
                    chosen_label = 22
                else:
                    chosen_label = 22
            else:
                chosen_label = 22
        except Exception as e:
            chosen_label = 22

        chosen_labels.append(chosen_label)
        if f"MOS_A_{eval_dim_}" in example and f"MOS_B_{eval_dim_}" in example:
            try:
                A_score = example[f"MOS_A_{eval_dim_}"] if example[f"MOS_A_{eval_dim_}"] is not None else 0.0
                B_score = example[f"MOS_B_{eval_dim_}"] if example[f"MOS_B_{eval_dim_}"] is not None else 0.0
            except Exception as e:
                A_score = 0.0
                B_score = 0.0
            A_scores.append(A_score)
            B_scores.append(B_score)
        else:
            A_scores.append(0.0)
            B_scores.append(0.0)

    chosen_labels = torch.tensor(chosen_labels, dtype=torch.long)
    A_scores = torch.tensor(A_scores, dtype=torch.float)
    B_scores = torch.tensor(B_scores, dtype=torch.float)
    metainfo_idx = None
    if 'metainfo_idx' in example:
        metainfo_idx = example['metainfo_idx']

    return {"A_data": A_data, "B_data": B_data,
            "A_scores": A_scores, "B_scores": B_scores,
            "chosen_label": chosen_labels,
            "metainfo_idx": metainfo_idx}
