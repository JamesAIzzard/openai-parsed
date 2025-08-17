from typing import Optional


def bool_parser(*, response: str) -> Optional[bool]:
    """Allows any valid boolean string representation.
    Allowed: "true", "false", "1", "0"
    Not Allowed: Everything else
    """
    response = response.strip().lower()
    if response in {"true", "1"}:
        return True
    elif response in {"false", "0"}:
        return False
    return None
