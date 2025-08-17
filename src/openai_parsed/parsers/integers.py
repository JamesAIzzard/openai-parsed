from typing import Optional

def int_parser(*, response: str) -> Optional[int]:
    """Allows any valid integer string representation."""
    response = response.strip()

    try:
        return int(response)
    except (ValueError, TypeError):
        return None