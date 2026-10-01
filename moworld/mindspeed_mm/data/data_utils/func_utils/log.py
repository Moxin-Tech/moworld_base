# Copyright 2025 Optuna, HuggingFace Inc. and the LlamaFactory team

import logging
import os
from functools import lru_cache


def get_logger(name):
    _logger = logging.getLogger(name)
    # Set the log level
    _logger.setLevel("INFO")
    # Set the log format
    formatter = logging.Formatter(
        '%(asctime)s - %(module)s - %(lineno)d - %(levelname)s - %(message)s')
    # Console print
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    _logger.addHandler(console_handler)
    return _logger


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


@lru_cache(None)
def warning_rank0_once(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0
logging.Logger.warning_rank0 = warning_rank0
logging.Logger.warning_rank0_once = warning_rank0_once
