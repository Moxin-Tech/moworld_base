# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.

import inspect
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Optional, TypedDict, Union, BinaryIO, Literal, List, Dict, Tuple, TypeVar, Any

import av
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageObject
from typing_extensions import override

VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")
IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")
AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")
IGNORE_INDEX = -100

# support newer transformers for qwen3omni
try:
    from transformers.video_processing_utils import BaseVideoProcessor
    has_video_proc = True
    VideoProcessorType = TypeVar('VideoProcessorType', BaseVideoProcessor, Any)
except ImportError:
    from transformers.image_processing_utils import BaseImageProcessor
    has_video_proc = False
    VideoProcessorType = TypeVar('VideoProcessorType', BaseImageProcessor, Any)

if TYPE_CHECKING:
    from av.stream import Stream
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor
    from transformers.image_utils import get_image_size, is_valid_image, to_numpy_array
    from numpy.typing import NDArray
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor


    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]


    ImageInput = Union[str, EncodedImage, ImageObject]
    VideoInput = str
    AudioInput = Union[str, BinaryIO, NDArray]


    class MMProcessor(ProcessorMixin):
        patch_size: int
        image_seq_length: int
        num_additional_image_tokens: int
        vision_feature_select_strategy: Literal["default", "full"]

        def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
            pass


def _make_batched_images(images: List["ImageObject"], imglens: List[int]) -> List[List["ImageObject"]]:
    r"""Make nested list of images."""
    batch_images = []
    for imglen in imglens:
        batch_images.append(images[:imglen])
        images = images[imglen:]

    return batch_images


def _check_video_is_nested_images(video: "VideoInput") -> bool:
    r"""Check if the video is nested images."""
    return isinstance(video, list) and all(isinstance(frame, (str, BinaryIO, dict, ImageObject)) for frame in video)


def get_video_processor(processor: Optional["MMProcessor"], base_model: bool = True) -> VideoProcessorType:
    r"""support newer transformers for qwen3omni."""
    if has_video_proc:
        return getattr(processor, "video_processor", None)
    if base_model:
        return getattr(processor, "video_processor", getattr(processor, "image_processor", None))
    return getattr(processor, "image_processor", None)


@dataclass
class MMPluginMixin:
    image_token: Optional[str]
    video_token: Optional[str]
    audio_token: Optional[str]
    expand_mm_tokens: bool = True

    def _validate_input(
            self,
            processor: Optional["MMProcessor"],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
    ) -> None:
        r"""Validate if this model accepts the input modalities."""
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor = get_video_processor(processor)
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct `template` is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct `template` is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct `template` is used."
            )

        if self.image_token is not None and processor is None:
            raise ValueError("Processor was not found, please check and update your model file.")

        if self.image_token is not None and image_processor is None:
            raise ValueError("Image processor was not found, please check and update your model file.")

        if self.video_token is not None and video_processor is None:
            raise ValueError("Video processor was not found, please check and update your model file.")

        if self.audio_token is not None and feature_extractor is None:
            raise ValueError("Audio feature extractor was not found, please check and update your model file.")

    @staticmethod
    def _validate_messages(
            messages: List[Dict[str, str]],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
    ):
        r"""Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        for message in messages:
            num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
            num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
            num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)

        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
            )

        if len(videos) != num_video_tokens:
            raise ValueError(
                f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
            )

        if len(audios) != num_audio_tokens:
            raise ValueError(
                f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
            )

    @staticmethod
    def _preprocess_image(
            image: "ImageObject", image_max_pixels: int, image_min_pixels: int, resample=True, **kwargs
    ) -> "ImageObject":
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            if resample:
                image = image.resize((width, height), resample=Image.Resampling.NEAREST)
            else:
                image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            if resample:
                image = image.resize((width, height), resample=Image.Resampling.NEAREST)
            else:
                image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Using the `with` statement ensures that the image file is automatically closed after the
        # code block is executed, effectively preventing resource leaks.
        # However, this introduces a problem: the `image` object directly obtained from
        # `Image.open()` is a `PIL.xxxImagePlugin` instance, which maintains an
        # association with the file on disk.

        # Once the `with` context is exited, the file handle is closed, and the `PIL.xxxImagePlugin`
        # object becomes invalid. At this point, any operation that requires re-reading the
        # file's content (such as an `np.array()` conversion or a `.save()` call) will fail
        # because the file is closed.

        # To solve this problem, we use the `.copy()` method.
        # `.copy()` loads all pixel data of the image completely into memory and creates a new,
        # independent image object. This new object is an instance of the `PIL.Image.Image`
        # class. It no longer depends on the original file and can therefore be safely
        # accessed and manipulated outside the `with` context.
        if image.__class__ != ImageObject:
            image = image.copy()

        return image

    @staticmethod
    def _get_video_sample_indices(
            video_stream: "Stream", video_fps: float, video_maxlen: int, **kwargs
    ) -> List[int]:
        r"""Compute video sample indices according to fps."""
        total_frames = video_stream.frames
        if total_frames == 0:  # infinite video
            return np.linspace(0, video_maxlen - 1, video_maxlen).astype(np.int32)

        sample_frames = max(1, math.floor(float(video_stream.duration * video_stream.time_base) * video_fps))
        sample_frames = min(total_frames, video_maxlen, sample_frames)
        return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32)

    def _regularize_images(self, images: List["ImageInput"], **kwargs) -> Dict[str, List["ImageObject"]]:
        r"""Regularize images to avoid error. Including reading and pre-processing."""
        results = []
        for image in images:
            processed_image = None

            if isinstance(image, (str, BinaryIO)):
                with Image.open(image) as image_obj:
                    processed_image = self._preprocess_image(image_obj, **kwargs)
            elif isinstance(image, bytes):
                with Image.open(BytesIO(image)) as image_obj:
                    processed_image = self._preprocess_image(image_obj, **kwargs)
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    with Image.open(BytesIO(image["bytes"])) as image_obj:
                        processed_image = self._preprocess_image(image_obj, **kwargs)
                else:
                    with Image.open(image["path"]) as image_obj:
                        processed_image = self._preprocess_image(image_obj, **kwargs)
            elif isinstance(image, ImageObject):
                processed_image = self._preprocess_image(image, **kwargs)

            if not isinstance(processed_image, ImageObject):
                raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

            results.append(processed_image)

        return {"images": results}

    def _regularize_videos(self, videos: List["VideoInput"], **kwargs) -> Dict[str, List[List["ImageObject"]]]:
        r"""Regularizes videos to avoid error. Including reading, resizing and converting."""
        results = []
        for video in videos:
            with av.open(video, "r") as container:
                video_stream = next(stream for stream in container.streams if stream.type == "video")
                sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                frames: List[ImageObject] = []
                container.seek(0)
                for frame_idx, frame in enumerate(container.decode(video_stream)):
                    if frame_idx in sample_indices:
                        frames.append(frame.to_image())

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results}

    @staticmethod
    def _regularize_audios(
            audios: List["AudioInput"], sampling_rate: float, **kwargs
    ) -> Dict[str, Union[List["NDArray"], List[float]]]:
        r"""Regularizes audios to avoid error. Including reading and resampling."""
        import librosa
        results, sampling_rates = [], []
        for audio in audios:
            if isinstance(audio, (str, BinaryIO)):
                audio, sampling_rate = librosa.load(audio, sr=sampling_rate)

            if not isinstance(audio, np.ndarray):
                raise ValueError(f"Expect input is a list of audios, but got {type(audio)}.")

            results.append(audio)
            sampling_rates.append(sampling_rate)

        return {"audios": results, "sampling_rates": sampling_rates}

    def _get_mm_inputs(
            self,
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            processor: "MMProcessor",
            imglens: Optional[List[int]] = None,
    ) -> Dict[str, "torch.Tensor"]:
        r"""Process visual inputs.

        Returns: (llava and paligemma)
            pixel_values: tensor with shape (B, C, H, W)

        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height
                            where num_patches == torch.prod(image_grid_thw)

        Returns: (mllama)
            pixel_values: tensor with shape
                          (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width)
                          For example, (2, 1, 4, 3, 560, 560).
            aspect_ratio_ids: tensor with shape (batch_size, max_num_images). For example, (2, 1).
            aspect_ratio_mask: tensor with shape (batch_size, max_num_images, max_image_tiles). For example, (2, 1, 4).
            num_tiles: List[List[int]] with shape (batch_size, num_images_in_batch). For example, (2, 1).

        """
        mm_inputs = {}
        if len(images) != 0:
            image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            image_processor_kwargs = {}
            if getattr(processor, "image_do_pan_and_scan", False):  # gemma3 image processor
                image_processor_kwargs.update(
                    {
                        "do_pan_and_scan": True,
                        "pan_and_scan_min_crop_size": 256,
                        "pan_and_scan_max_num_crops": 4,
                        "pan_and_scan_min_ratio_to_activate": 1.2,
                    }
                )

            mm_inputs.update(image_processor(images, return_tensors="pt", **image_processor_kwargs))

        if len(videos) != 0:
            video_processor = get_video_processor(processor)
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )["videos"]
            if "videos" in inspect.signature(video_processor.preprocess).parameters:  # for qwen2_vl and video_llava
                mm_inputs.update(video_processor(images=None, videos=videos, return_tensors="pt"))
            else:  # for llava_next_video
                mm_inputs.update(video_processor(videos, return_tensors="pt"))

        if len(audios) != 0:
            feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs


class BasePlugin(MMPluginMixin):
    def process_messages(
            self,
            messages: List[Dict[str, str]],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            processor: Optional["MMProcessor"],
    ) -> List[Dict[str, str]]:
        r"""Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
            self,
            input_ids: List[int],
            labels: Optional[List[int]],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            tokenizer: "PreTrainedTokenizer",
            processor: Optional["MMProcessor"],
    ) -> Tuple[List[int], Optional[List[int]]]:
        r"""Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
            self,
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            imglens: List[int],
            vidlens: List[int],
            audlens: List[int],
            batch_ids: List[List[int]],
            processor: Optional["MMProcessor"],
    ) -> Dict[str, Union[List[int], "torch.Tensor"]]:
        r"""Build batched multimodal inputs for VLMs.

        Arguments:
            images: a list of image inputs, shape (num_images,)
            videos: a list of video inputs, shape (num_videos,)
            audios: a list of audio inputs, shape (num_audios,)
            imglens: number of images in each sample, shape (batch_size,)
            vidlens: number of videos in each sample, shape (batch_size,)
            audlens: number of audios in each sample, shape (batch_size,)
            batch_ids: token ids of input samples, shape (batch_size, seq_len)
            processor: a processor for pre-processing images and videos

        """
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class Qwen2VLPlugin(BasePlugin):
    vision_bos_token: str = "<|vision_start|>"
    vision_eos_token: str = "<|vision_end|>"

    @override
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height))

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height))

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height))

        return image

    @override
    def _regularize_videos(
            self, videos: List["VideoInput"], **kwargs
    ) -> Dict[str, Union[List[List["ImageObject"]], List[float]]]:
        results, fps_per_video = [], []
        for video in videos:
            frames: List[ImageObject] = []
            if _check_video_is_nested_images(video):
                for frame in video:
                    if not is_valid_image(frame) and not isinstance(frame, dict) and not os.path.exists(frame):
                        raise ValueError("Invalid image found in video frames.")

                frames = video
                fps_per_video.append(kwargs.get("video_fps", 2.0))
            else:
                with av.open(video, "r") as container:
                    video_stream = next(stream for stream in container.streams if stream.type == "video")
                    sample_indices = self._get_video_sample_indices(video_stream, **kwargs)
                    container.seek(0)
                    for frame_idx, frame in enumerate(container.decode(video_stream)):
                        if frame_idx in sample_indices:
                            frames.append(frame.to_image())

                    if video_stream.duration is None:
                        fps_per_video.append(kwargs.get("video_fps", 2.0))
                    else:
                        fps_per_video.append(len(sample_indices) / float(video_stream.duration * video_stream.time_base))

            if len(frames) % 2 != 0:
                frames.append(frames[-1])

            frames = self._regularize_images(frames, **kwargs)["images"]
            results.append(frames)

        return {"videos": results, "fps_per_video": fps_per_video}

    @override
    def _get_mm_inputs(
            self,
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            processor: "MMProcessor",
    ) -> Dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            video_processor = get_video_processor(processor, False)
            mm_inputs.update(video_processor(videos=video_data["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in video_data["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
            self,
            messages: List[Dict[str, str]],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            processor: Optional["MMProcessor"],
    ) -> List[Dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_seqlen = video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    VIDEO_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Qwen3VLPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images: List["ImageInput"],
        videos: List["VideoInput"],
        audios: List["AudioInput"],
        processor: "MMProcessor",
    ) -> Dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
                resample=False,
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            videos = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            video_metadata = [
                {"fps": getattr(processor, "video_fps", 24.0), "duration": len(video), "total_num_frames": len(video)}
                for video in videos["videos"]
            ]
            mm_inputs.update(
                video_processor(videos=videos["videos"], video_metadata=video_metadata, return_metadata=True)
            )
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            if "second_per_grid_ts" in processor.model_input_names:
                mm_inputs["second_per_grid_ts"] = [temporal_patch_size / fps for fps in videos["fps_per_video"]]

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: List[Dict[str, str]],
        images: List["ImageInput"],
        videos: List["VideoInput"],
        audios: List["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> List[Dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")
        video_processor: BaseImageProcessor = getattr(processor, "video_processor")

        image_merge_length: int = getattr(image_processor, "merge_size") ** 2
        video_merge_length: int = getattr(video_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            num_frames = video_grid_thw[0][0] if len(video_grid_thw) > 0 else 0  # hard code for now
            video_metadata = mm_inputs.get("video_metadata", {})

        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            num_frames = 0
            timestamps = [0]

        for idx, message in enumerate(messages):
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod() // image_merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                metadata = video_metadata[idx]
                timestamps = processor._calculate_timestamps(
                    metadata.frames_indices,
                    metadata.fps,
                    video_processor.merge_size,
                )
                video_structure = ""
                for frame_index in range(num_frames):
                    video_seqlen = (
                        video_grid_thw[num_video_tokens][1:].prod() // video_merge_length
                        if self.expand_mm_tokens
                        else 1
                    )
                    timestamp_sec = timestamps[frame_index]
                    frame_structure = (
                        f"<{timestamp_sec:.1f} seconds>"
                        f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}"
                    )
                    video_structure += frame_structure

                if not self.expand_mm_tokens:
                    video_structure = f"{self.vision_bos_token}{self.video_token}{self.vision_eos_token}"

                content = content.replace(VIDEO_PLACEHOLDER, video_structure, 1)
                num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class GLM4VPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
    ) -> dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        video_processor: BaseImageProcessor = getattr(processor, "video_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
                resample=False,
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_data = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            # prepare video metadata
            video_metadata = [
                {
                    "fps": 2,
                    "duration": len(video),
                    "total_frames": len(video)
                }
                for video in video_data["videos"]
            ]
            mm_inputs.update(video_processor(images=None, videos=video_data["videos"], video_metadata=video_metadata))

        return mm_inputs

    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens = 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            num_frames = video_grid_thw[0][0] if len(video_grid_thw) > 0 else 0  # hard code for now
            timestamps = mm_inputs.get("timestamps", [])

            if hasattr(timestamps, "tolist"):
                timestamps = timestamps.tolist()

            if not timestamps:
                timestamps_list = []
            elif isinstance(timestamps[0], list):
                timestamps_list = timestamps[0]
            else:
                timestamps_list = timestamps

            unique_timestamps = timestamps_list.copy()
            selected_timestamps = unique_timestamps[:num_frames]
            while len(selected_timestamps) < num_frames:
                selected_timestamps.append(selected_timestamps[-1] if selected_timestamps else 0)

        else:
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            num_frames = 0
            selected_timestamps = [0]

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|begin_of_image|>{self.image_token * image_seqlen}<|end_of_image|>", 1
                )
                num_image_tokens += 1

            while VIDEO_PLACEHOLDER in content:
                video_structure = ""
                for frame_index in range(num_frames):
                    video_seqlen = (
                        video_grid_thw[num_video_tokens][1:].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    timestamp_sec = selected_timestamps[frame_index]
                    frame_structure = (
                        f"<|begin_of_image|>{self.image_token * video_seqlen}<|end_of_image|>{timestamp_sec}"
                    )
                    video_structure += frame_structure

                if not self.expand_mm_tokens:
                    video_structure = self.video_token

                content = content.replace(VIDEO_PLACEHOLDER, f"<|begin_of_video|>{video_structure}<|end_of_video|>", 1)
                num_video_tokens += 1

            message["content"] = content

        return messages

    @override
    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["ProcessorMixin"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        self._validate_input(processor, images, videos, audios)
        mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
        mm_inputs.pop("timestamps", None)
        return mm_inputs


class Qwen2OmniPlugin(Qwen2VLPlugin):
    @override
    def _get_mm_inputs(
            self,
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            processor: "MMProcessor",
    ) -> Dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
                resample=False,
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_dict = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            video_processor = get_video_processor(processor)
            mm_inputs.update(video_processor(videos=video_dict["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            mm_inputs["video_second_per_grid"] = torch.tensor(
                [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
            )

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs

    @override
    def process_messages(
            self,
            messages: List[Dict[str, str]],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            processor: Optional["MMProcessor"],
    ) -> List[Dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)

        merge_length = processor.image_processor.merge_size ** 2
        use_audio_in_video = getattr(processor, "use_audio_in_video", False)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            if "feature_attention_mask" in mm_inputs:
                input_lengths = (mm_inputs["feature_attention_mask"].sum(-1).numpy() - 1) // 2 + 1
                audio_lengths = (input_lengths - 2) // 2 + 1
        else:
            mm_inputs = {}
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            audio_lengths = [None] * len(audios)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER, f"<|vision_bos|>{self.image_token * image_seqlen}<|vision_eos|>", 1
                )
                num_image_tokens += 1

            if (
                    use_audio_in_video and len(audios) and len(videos)
            ):  # if use the audio of video # deal video token and audio token together
                if len(videos) != len(audios):
                    raise ValueError(
                        f"Number of videos ({len(videos)}) must match number of audios ({len(audios)}) when using audio in video."
                    )

                while VIDEO_PLACEHOLDER in content:
                    video_pos = content.find(VIDEO_PLACEHOLDER)
                    audio_pos = content.find(AUDIO_PLACEHOLDER, video_pos)
                    if audio_pos == -1 or audio_pos < video_pos:
                        raise ValueError(
                            f"Each {VIDEO_PLACEHOLDER} must be followed by an {AUDIO_PLACEHOLDER} when using audio in video."
                        )

                    audio_t_index = torch.arange(audio_lengths[num_audio_tokens])
                    video_t_index = (
                            torch.arange(video_grid_thw[num_video_tokens][0])
                            .view(-1, 1, 1)
                            .expand(
                                -1,
                                video_grid_thw[num_video_tokens][1] // image_processor.merge_size,
                                video_grid_thw[num_video_tokens][2] // image_processor.merge_size,
                            )
                            .flatten()
                            * mm_inputs.get("video_second_per_grid").get(num_video_tokens)
                            * 25
                    ).long()
                    t_ntoken_per_chunk = 50
                    video_chunk_indices = processor.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                    audio_chunk_indices = processor.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                    placeholder_string = ""
                    placeholder_string += "<|vision_bos|>" + "<|audio_bos|>"
                    for j in range(max(len(video_chunk_indices), len(audio_chunk_indices))):
                        video_chunk_index = video_chunk_indices[j] if j < len(video_chunk_indices) else None
                        audio_chunk_index = audio_chunk_indices[j] if j < len(audio_chunk_indices) else None
                        if video_chunk_index is not None:
                            placeholder_string += self.video_token * (video_chunk_index[1] - video_chunk_index[0])

                        if audio_chunk_index is not None:
                            placeholder_string += self.audio_token * (audio_chunk_index[1] - audio_chunk_index[0])

                    placeholder_string += "<|audio_eos|>" + "<|vision_eos|>"
                    content = content.replace(VIDEO_PLACEHOLDER, placeholder_string, 1)
                    content = content.replace(AUDIO_PLACEHOLDER, "", 1)
                    num_audio_tokens += 1
                    num_video_tokens += 1
            else:
                while AUDIO_PLACEHOLDER in content:
                    audio_seqlen = audio_lengths[num_audio_tokens] if self.expand_mm_tokens else 1
                    content = content.replace(
                        AUDIO_PLACEHOLDER, f"<|audio_bos|>{self.audio_token * audio_seqlen}<|audio_eos|>", 1
                    )
                    num_audio_tokens += 1

                while VIDEO_PLACEHOLDER in content:
                    video_seqlen = (
                        video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    content = content.replace(
                        VIDEO_PLACEHOLDER, f"<|vision_bos|>{self.video_token * video_seqlen}<|vision_eos|>", 1
                    )
                    num_video_tokens += 1

            message["content"] = content

        return messages


@dataclass
class Qwen3OmniPlugin(Qwen2VLPlugin):
    audio_bos_token: str = "<|audio_start|>"
    audio_eos_token: str = "<|audio_end|>"

    @override
    def _get_mm_inputs(
            self,
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            processor: "MMProcessor",
    ) -> Dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        feature_extractor: SequenceFeatureExtractor = getattr(processor, "feature_extractor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
                resample=False,
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        if len(videos) != 0:
            video_dict = self._regularize_videos(
                videos,
                image_max_pixels=getattr(processor, "video_max_pixels", 256 * 256),
                image_min_pixels=getattr(processor, "video_min_pixels", 16 * 16),
                video_fps=getattr(processor, "video_fps", 2.0),
                video_maxlen=getattr(processor, "video_maxlen", 128),
            )
            video_processor = get_video_processor(processor, False)
            mm_inputs.update(video_processor(videos=video_dict["videos"], return_tensors="pt"))
            temporal_patch_size: int = getattr(image_processor, "temporal_patch_size", 2)
            mm_inputs["video_second_per_grid"] = torch.tensor(
                [temporal_patch_size / fps for fps in video_dict["fps_per_video"]]
            )

        if len(audios) != 0:
            audios = self._regularize_audios(
                audios,
                sampling_rate=getattr(processor, "audio_sampling_rate", 16000),
            )["audios"]
            if getattr(processor, "audio_sampling_rate", 16000) != feature_extractor.sampling_rate:
                import librosa
                resampled_audios = []
                for audio in audios:
                    resampled_audio = librosa.resample(audio, orig_sr=getattr(processor, "audio_sampling_rate", 16000), target_sr=feature_extractor.sampling_rate)
                    resampled_audios.append(resampled_audio)
                audios = resampled_audios

            mm_inputs.update(
                feature_extractor(
                    audios,
                    sampling_rate=feature_extractor.sampling_rate,
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
            )
            mm_inputs["feature_attention_mask"] = mm_inputs.pop("attention_mask")  # prevent conflicts

        return mm_inputs

    @override
    def process_messages(
            self,
            messages: List[Dict[str, str]],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            processor: Optional["MMProcessor"],
    ) -> List[Dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        messages = deepcopy(messages)
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)

        merge_length = processor.image_processor.merge_size ** 2
        use_audio_in_video = getattr(processor, "use_audio_in_video", False)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("image_grid_thw", [])
            video_grid_thw = mm_inputs.get("video_grid_thw", [])
            if "feature_attention_mask" in mm_inputs:
                input_lengths = mm_inputs["feature_attention_mask"].sum(-1)
                input_lengths_leave = input_lengths % 100
                feature_lengths = (input_lengths_leave - 1) // 2 + 1
                audio_lengths = ((feature_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        else:
            mm_inputs = {}
            image_grid_thw = [None] * len(images)
            video_grid_thw = [None] * len(videos)
            audio_lengths = [None] * len(audios)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = image_grid_thw[num_image_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"{self.vision_bos_token}{self.image_token * image_seqlen}{self.vision_eos_token}",
                    1,
                )
                num_image_tokens += 1

            if (
                    use_audio_in_video and len(audios) and len(videos)
            ):  # if use the audio of video # deal video token and audio token together
                if len(videos) != len(audios):
                    raise ValueError(
                        f"Number of videos ({len(videos)}) must match number of audios ({len(audios)}) when using audio in video."
                    )

                while VIDEO_PLACEHOLDER in content:
                    video_pos = content.find(VIDEO_PLACEHOLDER)
                    audio_pos = content.find(AUDIO_PLACEHOLDER, video_pos)
                    if audio_pos == -1 or audio_pos < video_pos:
                        raise ValueError(
                            f"Each {VIDEO_PLACEHOLDER} must be followed by an {AUDIO_PLACEHOLDER} when using audio in video."
                        )

                    audio_t_index = torch.arange(audio_lengths[num_audio_tokens])
                    video_t_index = (
                            torch.arange(video_grid_thw[num_video_tokens][0])
                            .view(-1, 1, 1)
                            .expand(
                                -1,
                                video_grid_thw[num_video_tokens][1] // image_processor.merge_size,
                                video_grid_thw[num_video_tokens][2] // image_processor.merge_size,
                            )
                            .flatten()
                            * mm_inputs.get("video_second_per_grid")[num_video_tokens]
                            * 25
                    ).long()
                    t_ntoken_per_chunk = 50
                    video_chunk_indices = processor.get_chunked_index(video_t_index, t_ntoken_per_chunk)
                    audio_chunk_indices = processor.get_chunked_index(audio_t_index, t_ntoken_per_chunk)
                    placeholder_string = ""
                    placeholder_string += self.vision_bos_token + self.audio_bos_token
                    for j in range(max(len(video_chunk_indices), len(audio_chunk_indices))):
                        video_chunk_index = video_chunk_indices[j] if j < len(video_chunk_indices) else None
                        audio_chunk_index = audio_chunk_indices[j] if j < len(audio_chunk_indices) else None
                        if video_chunk_index is not None:
                            placeholder_string += self.video_token * (video_chunk_index[1] - video_chunk_index[0])

                        if audio_chunk_index is not None:
                            placeholder_string += self.audio_token * (audio_chunk_index[1] - audio_chunk_index[0])

                    placeholder_string += self.audio_eos_token + self.vision_eos_token
                    content = content.replace(VIDEO_PLACEHOLDER, placeholder_string, 1)
                    content = content.replace(AUDIO_PLACEHOLDER, "", 1)
                    num_audio_tokens += 1
                    num_video_tokens += 1
            else:
                while AUDIO_PLACEHOLDER in content:
                    audio_seqlen = audio_lengths[num_audio_tokens] if self.expand_mm_tokens else 1
                    content = content.replace(
                        AUDIO_PLACEHOLDER,
                        f"{self.audio_bos_token}{self.audio_token * audio_seqlen}{self.audio_eos_token}",
                        1,
                    )
                    num_audio_tokens += 1

                while VIDEO_PLACEHOLDER in content:
                    video_seqlen = (
                        video_grid_thw[num_video_tokens].prod() // merge_length if self.expand_mm_tokens else 1
                    )
                    content = content.replace(
                        VIDEO_PLACEHOLDER,
                        f"{self.vision_bos_token}{self.video_token * video_seqlen}{self.vision_eos_token}",
                        1,
                    )
                    num_video_tokens += 1

            message["content"] = content

        return messages


class KimiK25Plugin(BasePlugin):
    @override
    def _get_mm_inputs(
        self,
        images: List["ImageInput"],
        videos: List["VideoInput"],
        audios: List["AudioInput"],
        processor: "MMProcessor",
    ) -> Dict[str, "torch.Tensor"]:
        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        mm_inputs = {}
        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            mm_inputs.update(image_processor(images, return_tensors="pt"))

        return mm_inputs

    @override
    def process_messages(self, messages, images, videos, audios, processor):
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)

        num_image_tokens = 0
        messages = deepcopy(messages)

        image_processor: BaseImageProcessor = getattr(processor, "image_processor", None)
        image_merge_length: int = getattr(image_processor, "media_proc_cfg")["merge_kernel_size"] ** 2

        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            image_grid_thw = mm_inputs.get("grid_thws", [])
        else:
            image_grid_thw = [None] * len(images)

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                image_seqlen = (
                    image_grid_thw[num_image_tokens].prod() // image_merge_length if self.expand_mm_tokens else 1
                )
                content = content.replace(
                    IMAGE_PLACEHOLDER,
                    f"<|media_start|>image<|media_content|>{self.image_token * image_seqlen}<|media_end|>",
                    1,
                )
                num_image_tokens += 1

            message["content"] = content

        return messages


PLUGINS = {
    "base": BasePlugin,
    "qwen2_vl": Qwen2VLPlugin,
    "qwen2_omni": Qwen2OmniPlugin,
    "qwen3_vl": Qwen3VLPlugin,
    "glm4.1v": GLM4VPlugin,
    "qwen3_omni": Qwen3OmniPlugin,
    "kimi_k25": KimiK25Plugin,
}


def get_mm_plugin(
        name: str,
        image_token: Optional[str] = None,
        video_token: Optional[str] = None,
        audio_token: Optional[str] = None,
) -> "BasePlugin":
    r"""Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token)
