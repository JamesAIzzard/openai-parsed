from typing import Optional


def string_parser(*, response: str) -> Optional[str]:
    """Allows any non-empty string response."""
    response = response.strip()
    return response if response else None
