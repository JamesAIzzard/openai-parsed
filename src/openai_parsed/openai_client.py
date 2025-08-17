from __future__ import annotations
from typing import Optional, Protocol, TypeVar
import os
import logging

import openai
from rich.console import Console

from codiet.exceptions.db_population import AIDeclinedError, ExhaustedGenRetriesError

T = TypeVar("T", covariant=True)


class Parser(Protocol[T]):
    def __call__(self, *, response: str) -> Optional[T]: ...


def _get_response(input: str) -> str:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("Missing OPENAI_API_KEY environment variable")

    response = openai.responses.create(
        model="gpt-4.1-mini",
        input=input,
    )
    return response.output_text


def _add_common_preface(user_input: str) -> str:
    return (
        "You are assisting data collection for a meal-planning application.\n"
        "Output plain ASCII text—no markdown, styling, or Unicode escapes.\n"
        + user_input
    )


def ask_and_validate(
    *,
    prompt: str,
    parser: Parser[T],
    console: Console,
    max_retries: int = 10,
) -> T:
    prompt = _add_common_preface(prompt)

    response_log = {}

    for attempt in range(1, max_retries + 1):
        logging.debug(f"Prompt:\n{prompt}")
        raw = _get_response(prompt)
        logging.debug(f"Response:\n{raw}")
        response_log[attempt] = raw

        if raw.strip().lower() == "declined":
            logging.debug("Model declined response.")
            raise AIDeclinedError()

        parsed = parser(response=raw)

        if parsed is not None:
            return parsed

        console.log(
            f"[yellow]Invalid response: '{raw}'. Retrying... ({attempt}/{max_retries})"
        )
        console.log(f"Retrying… ({attempt}/{max_retries})")

    raise ExhaustedGenRetriesError(
        max_retries=max_retries, prompt=prompt, response_log=response_log
    )
