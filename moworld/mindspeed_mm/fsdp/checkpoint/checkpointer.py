from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union


class CheckpointerBase(ABC):
    """Base class for checkpointer"""

    @classmethod
    @abstractmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: Optional[bool],
        iteration: Optional[int],
    ):
        return

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
    ):
        return
