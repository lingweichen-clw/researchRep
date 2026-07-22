"""Persistent block-memory storage and construction."""

from typing import Any

from .schema import BankManifest
from .storage import BankWriter, CalendarIndex, MemoryBank

__all__ = ["BankManifest", "BankWriter", "CalendarIndex", "MemoryBank", "build_memory_bank"]


def __getattr__(name: str) -> Any:
    if name == "build_memory_bank":
        from .builder import build_memory_bank

        return build_memory_bank
    raise AttributeError(name)
