from __future__ import annotations

from openai_parsed.parsers.booleans import bool_parser
from openai_parsed.parsers.floats import float_parser
from openai_parsed.parsers.integers import int_parser
from openai_parsed.parsers.strings import string_parser


class TestStringParser:
    def test_accepts_non_empty_string(self) -> None:
        """Verify returns stripped string when non-empty."""
        assert string_parser(response="  hello ") == "hello"

    def test_rejects_empty_or_whitespace(self) -> None:
        """Verify returns None for empty or whitespace-only strings."""
        assert string_parser(response="") is None
        assert string_parser(response="   ") is None


class TestIntParser:
    def test_accepts_valid_integer(self) -> None:
        """Verify parses valid integer representations."""
        assert int_parser(response="42") == 42
        assert int_parser(response="  -7 ") == -7

    def test_rejects_invalid_integer(self) -> None:
        """Verify returns None for invalid integer strings."""
        assert int_parser(response="4.2") is None
        assert int_parser(response="abc") is None


class TestFloatParser:
    def test_accepts_valid_float(self) -> None:
        """Verify parses valid float representations including comma decimal separator."""
        assert float_parser(response="3.14") == 3.14
        assert float_parser(response="  -2,5 ") == -2.5

    def test_rejects_invalid_float(self) -> None:
        """Verify returns None for invalid float strings."""
        assert float_parser(response="abc") is None
        assert float_parser(response="") is None


class TestBoolParser:
    def test_accepts_true_variants(self) -> None:
        """Verify returns True for allowed true-like strings."""
        for val in ("true", "1", " TRUE "):
            assert bool_parser(response=val) is True

    def test_accepts_false_variants(self) -> None:
        """Verify returns False for allowed false-like strings."""
        for val in ("false", "0", " FALSE "):
            assert bool_parser(response=val) is False

    def test_rejects_invalid_bool(self) -> None:
        """Verify returns None for invalid boolean strings."""
        assert bool_parser(response="yes") is None
        assert bool_parser(response="") is None

