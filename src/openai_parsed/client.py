from __future__ import annotations
from typing import Optional, TypeVar
import os
import logging

import openai
from rich.console import Console
from rich import get_console

from .exceptions import LLMDeclinedError, LLMRetriesError, ParseFailedError
from .types import ParsedOpenAIClient, Parser

T = TypeVar("T", covariant=True)


class _ParsedOpenAIClient(ParsedOpenAIClient):
    def __init__(
        self,
        *,
        model: str,
        console: Optional[Console] = None,
        std_preface: Optional[str] = "",
        max_retries: int = 10,
        allow_decline: bool = True,
    ):
        self._model = model
        self._console = console or get_console()
        self._std_preface = std_preface
        self._max_retries = max_retries
        self._allow_decline = allow_decline
        self._allow_decline_message = (
            "If you don't have enough knowledge to provide a reasonable answer, "
            "respond only with the word 'DECLINED'\n"
        )

    def set_std_preface(self, preface: str) -> None:
        self._std_preface = preface

    def clear_std_preface(self) -> None:
        self._std_preface = ""

    def _get_response(self, prompt: str) -> str:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        response = openai.responses.create(
            model=self._model,
            input=prompt,
        )
        return response.output_text

    def ensure(
        self,
        *,
        prompt: str,
        parser: Parser[T],
        max_retries: Optional[int] = None,
        allow_decline: Optional[bool] = None,
    ) -> T:
        allow_decline = (
            allow_decline if allow_decline is not None else self._allow_decline
        )

        if self._std_preface is None:
            self._std_preface = ""
        prompt = self._std_preface + prompt

        if allow_decline:
            prompt += self._allow_decline_message

        response_log: dict[int, str] = {}
        max_retries = self._max_retries if max_retries is None else max_retries

        for attempt in range(1, max_retries + 1):
            logging.debug(f"Prompt:\n{prompt}")

            raw = self._get_response(prompt)

            logging.debug(f"Response:\n{raw}")

            response_log[attempt] = raw

            if allow_decline:
                if raw.strip().replace('"', "").replace("'", "").lower() == "declined":
                    logging.debug("Model declined response.")
                    raise LLMDeclinedError(prompt=prompt)

            try:
                parsed = parser(response=raw)
                return parsed
            except ParseFailedError:
                self._console.log(
                    f"[yellow]Invalid response: '{raw}'. Retrying... ({attempt}/{max_retries})"
                )

        raise LLMRetriesError(
            max_retries=max_retries, prompt=prompt, response_log=response_log
        )
