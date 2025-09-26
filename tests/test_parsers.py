from __future__ import annotations

import pytest

from openai_parsed.parsers import (
    parse_boolean,
    parse_float,
    parse_integer,
    StringChoiceParser,
)
from openai_parsed.exceptions import ParseFailedError


class TestParseBoolean:
    def test_true_and_yes(self) -> None:
        """Verify returns True for 'true' and 'yes' (case-insensitive)."""
        assert parse_boolean(response="true") is True
        assert parse_boolean(response="yes") is True
        assert parse_boolean(response="TrUe") is True
        assert parse_boolean(response="YeS") is True

    def test_false_and_no(self) -> None:
        """Verify returns False for 'false' and 'no' (case-insensitive)."""
        assert parse_boolean(response="false") is False
        assert parse_boolean(response="no") is False
        assert parse_boolean(response="FaLsE") is False
        assert parse_boolean(response="NO") is False

    def test_whitespace_is_ignored(self) -> None:
        """Verify strips leading and trailing whitespace before parsing."""
        assert parse_boolean(response="  yes \n") is True
        assert parse_boolean(response="\t no \t") is False

    def test_invalid_raises(self) -> None:
        """Verify raises ParseFailedError when input is unrecognised."""
        with pytest.raises(ParseFailedError):
            parse_boolean(response="maybe")


class TestParseFloat:
    def test_basic_and_scientific(self) -> None:
        """Verify parses standard and scientific-notation floats."""
        assert parse_float(response="3.14") == pytest.approx(3.14)
        assert parse_float(response="1e3") == pytest.approx(1000.0)
        assert parse_float(response="-2.5") == pytest.approx(-2.5)

    def test_whitespace_is_ignored(self) -> None:
        """Verify strips whitespace before float conversion."""
        assert parse_float(response="  42.0 \n") == pytest.approx(42.0)

    def test_invalid_raises(self) -> None:
        """Verify raises ParseFailedError when value is not a float."""
        with pytest.raises(ParseFailedError):
            parse_float(response="abc")
        with pytest.raises(ParseFailedError):
            parse_float(response="1.2.3")


class TestParseInteger:
    def test_basic_and_negative(self) -> None:
        """Verify parses basic and negative integers."""
        assert parse_integer(response="7") == 7
        assert parse_integer(response="-12") == -12

    def test_whitespace_is_ignored(self) -> None:
        """Verify strips whitespace before integer conversion."""
        assert parse_integer(response="  15 \t") == 15

    def test_invalid_raises(self) -> None:
        """Verify raises ParseFailedError when value is not an integer."""
        with pytest.raises(ParseFailedError):
            parse_integer(response="3.0")
        with pytest.raises(ParseFailedError):
            parse_integer(response="ten")


class TestStringChoiceParser:
    def test_returns_valid_choice(self) -> None:
        """Verify returns the string when it matches a valid choice."""
        parser = StringChoiceParser({"red", "green", "blue"})
        assert parser(response="red") == "red"

    def test_strips_whitespace(self) -> None:
        """Verify trims whitespace before validating membership."""
        parser = StringChoiceParser({"alpha", "beta"})
        assert parser(response="  beta \n") == "beta"

    def test_case_sensitivity(self) -> None:
        """Verify membership check is case-sensitive."""
        parser = StringChoiceParser({"yes", "no"})
        with pytest.raises(ParseFailedError):
            parser(response="Yes")

    def test_invalid_raises(self) -> None:
        """Verify raises ParseFailedError when value not in choices."""
        parser = StringChoiceParser({"cat", "dog"})
        with pytest.raises(ParseFailedError):
            parser(response="hamster")
