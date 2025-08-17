from __future__ import annotations

from typing import Callable

import pytest

from openai_parsed.client import ParsedOpenAIClient
from openai_parsed.exceptions import LLMDeclinedError, LLMRetriesError
from openai_parsed.parsers.strings import string_parser


class TestEnsureSuccess:
    def test_returns_parsed_value_first_try(self, make_client: Callable[..., ParsedOpenAIClient]) -> None:
        """Verify ensure returns parser output when first response is valid."""

        client = make_client()

        def fake_get_response(input: str) -> str:
            return "  hello world  "

        # Patch private method to avoid hitting the real API.
        client._get_response = fake_get_response

        result = client.ensure(prompt="say hi", parser=string_parser)
        assert result == "hello world"


class TestDeclineBehaviour:
    def test_raises_declined_when_model_declines(self, make_client: Callable[..., ParsedOpenAIClient]) -> None:
        """Verify ensure raises LLMDeclinedError when response is DECLINED and decline allowed."""

        client = make_client()

        def fake_get_response(input: str) -> str:
            # Return with quotes and varied case to test normalisation.
            return " 'DeClInEd' \n"

        client._get_response = fake_get_response

        with pytest.raises(LLMDeclinedError):
            client.ensure(prompt="anything", parser=string_parser)

    def test_no_decline_check_when_disabled(self, make_client: Callable[..., ParsedOpenAIClient]) -> None:
        """Verify DECLINED is treated as normal text when decline is disabled."""

        client = make_client()
        client._get_response = lambda input: " DECLINED "

        # Explicitly disable decline handling.
        result = client.ensure(prompt="anything", parser=string_parser, allow_decline=False)
        assert result == "DECLINED"


class TestRetryBehaviour:
    def test_retries_then_succeeds(self, make_client: Callable[..., ParsedOpenAIClient]) -> None:
        """Verify invalid responses trigger retries and eventual success returns value."""

        client = make_client()
        calls: list[str] = []

        def fake_get_response(input: str) -> str:
            calls.append(input)
            # First two attempts invalid (empty), third valid
            return {0: " ", 1: "\n\t", 2: "ok"}[len(calls) - 1]

        client._get_response = fake_get_response

        result = client.ensure(prompt="please reply", parser=string_parser, max_retries=3)
        assert result == "ok"
        assert len(calls) == 3

    def test_exhausts_retries_and_raises(self, make_client: Callable[..., ParsedOpenAIClient]) -> None:
        """Verify raises LLMRetriesError after max_retries invalid responses with log preserved."""

        client = make_client()

        def fake_get_response(input: str) -> str:
            return "   "  # Always invalid for string_parser

        client._get_response = fake_get_response

        with pytest.raises(LLMRetriesError) as exc:
            client.ensure(prompt="please reply", parser=string_parser, max_retries=3)

        assert exc.value.max_retries == 3
        assert isinstance(exc.value.response_log, dict)
        assert len(exc.value.response_log) == 3


class TestPromptComposition:
    def test_includes_preface_and_decline_message(self, make_client: Callable[..., ParsedOpenAIClient]) -> None:
        """Verify std preface and decline instruction are appended to prompt sent to LLM."""

        client = make_client(std_preface="[P] ")

        captured_prompt: list[str] = []

        def fake_get_response(input: str) -> str:
            captured_prompt.append(input)
            return "ok"

        client._get_response = fake_get_response

        # Use a parser that will immediately accept to avoid retries.
        result = client.ensure(prompt="Question?", parser=string_parser)
        assert result == "ok"

        assert len(captured_prompt) == 1
        sent = captured_prompt[0]
        assert sent.startswith("[P] Question?")
        assert "DECLINED" in sent  # decline instruction appended when allowed

    def test_no_decline_message_when_disabled(self, make_client: Callable[..., ParsedOpenAIClient]) -> None:
        """Verify decline instruction is omitted when allow_decline is False."""

        client = make_client(std_preface="[P] ")
        captured: list[str] = []

        def fake_get_response(input: str) -> str:
            captured.append(input)
            return "ok"

        client._get_response = fake_get_response

        _ = client.ensure(prompt="Question?", parser=string_parser, allow_decline=False)
        assert len(captured) == 1
        assert "DECLINED" not in captured[0]
