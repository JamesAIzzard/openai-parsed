from typing import Protocol, TypeVar, Optional

T = TypeVar("T", covariant=True)

class Parser(Protocol[T]):
    def __call__(self, *, response: str) -> Optional[T]: ...