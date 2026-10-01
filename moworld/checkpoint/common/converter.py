from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from pydantic import DirectoryPath


class Commandable:
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        """All subclasses of Converter will be stored in the class attribute 'subclalsses'"""
        super().__init_subclass__(**kwargs)
        own_abstract = any(getattr(value, '__isabstractmethod__', False) for key, value in cls.__dict__.items())
        if not own_abstract:
            cls.subclasses.append(cls)

    @classmethod
    def add_command(cls, command: Callable):
        cls.subclasses.append(command)


class Converter(ABC, Commandable):

    @staticmethod
    @abstractmethod
    def hf_to_mm(cfg):
        pass

    @staticmethod
    @abstractmethod
    def mm_to_hf(cfg):
        pass

    @staticmethod
    @abstractmethod
    def resplit(cfg):
        pass


class DcpConverter(ABC, Commandable):
    @staticmethod
    @abstractmethod
    def hf_to_dcp(hf_dir: DirectoryPath, save_dir: Path):
        pass

    @staticmethod
    @abstractmethod
    def dcp_to_hf(hf_dir: DirectoryPath, dcp_dir: DirectoryPath, save_dir: Path):
        pass
