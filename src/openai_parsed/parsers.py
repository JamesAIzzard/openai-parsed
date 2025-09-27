from __future__ import annotations

from .types import Parser
from .exceptions import ParseFailedError


def parse_boolean(*, response: str) -> bool:
    normalized = response.strip().lower()
    if normalized in {"true", "yes"}:
        return True
    if normalized in {"false", "no"}:
        return False
    raise ParseFailedError(response=response)


def parse_float(*, response: str) -> float:
    try:
        return float(response.strip())
    except ValueError:
        raise ParseFailedError(response=response)


def parse_integer(*, response: str) -> int:
    try:
        return int(response.strip())
    except ValueError:
        raise ParseFailedError(response=response)


class StringListParser(Parser[list[str]]):
    def __init__(self, *, separator: str = ",", allow_empty: bool = False):
        self._separator = separator
        self._allow_empty = allow_empty

    def __call__(self, *, response: str) -> list[str]:
        items = [
            item.strip() for item in response.split(self._separator) if item.strip()
        ]
        if not items and not self._allow_empty:
            raise ParseFailedError(response=response)
        return items


class StringChoiceParser(Parser[str]):
    def __init__(self, choices: set[str]):
        self.choices = choices

    def __call__(self, *, response: str) -> str:
        normalized = response.strip()
        if normalized in self.choices:
            return normalized
        raise ParseFailedError(response=response)
