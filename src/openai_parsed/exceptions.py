from typing import Mapping


class OpenAIParserError(Exception): ...


class LLMDeclinedError(OpenAIParserError):
    def __init__(self, prompt: str):
        self.prompt = prompt


class ParseFailedError(OpenAIParserError):
    def __init__(self, response: str):
        self.response = response


class LLMRetriesError(OpenAIParserError):
    def __init__(self, max_retries: int, prompt: str, response_log: Mapping[int, str]):
        self.max_retries = max_retries
        self.prompt = prompt
        self.response_log = response_log
