from __future__ import annotations

from typing import Callable, Protocol

import pytest

from openai_parsed.client import ParsedOpenAIClient


class _Console(Protocol):
    def log(self, message: str) -> None: ...


class DummyConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


@pytest.fixture
def dummy_console() -> DummyConsole:
    return DummyConsole()


@pytest.fixture
def make_client(dummy_console: _Console) -> Callable[..., ParsedOpenAIClient]:
    def _factory(**kwargs: object) -> ParsedOpenAIClient:
        return ParsedOpenAIClient(model="gpt-test", console=dummy_console, **kwargs)  # type: ignore[arg-type]

    return _factory

