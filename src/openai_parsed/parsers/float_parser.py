from typing import Optional


def float_parser(*, response: str) -> Optional[float]:
    """Allows any valid float string representation."""
    response = response.strip().replace(",", ".")

    try:
        return float(response)
    except (ValueError, TypeError):
        return None
