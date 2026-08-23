"""Unit tests for the shared text-hygiene helpers."""

from __future__ import annotations

from carelite.safety import normalize


def test_nfkc_folds_fullwidth_homoglyphs() -> None:
    assert normalize.normalize_text("\uff49\uff47\uff4e\uff4f\uff52\uff45") == "ignore"


def test_invisible_characters_are_detected_and_stripped() -> None:
    hostile = "ig\u200bnore"
    assert normalize.has_invisibles(hostile)
    assert normalize.strip_invisibles(hostile) == "ignore"
    assert not normalize.has_invisibles(normalize.normalize_text(hostile))


def test_control_characters_are_detected() -> None:
    assert normalize.has_control_chars("bad\x07text")
    assert "\x07" not in normalize.normalize_text("bad\x07text")


def test_detection_form_drops_punctuation_and_case() -> None:
    assert normalize.detection_form("Chest-Pain, severe!") == "chest pain severe"


def test_detection_form_joins_contractions() -> None:
    """ "can't" must become "cant" so one written phrase covers both spellings."""
    assert normalize.detection_form("I can't go on") == "i cant go on"


def test_detection_form_collapses_character_padding() -> None:
    assert normalize.detection_form("chesttttt paiiiin") == "chest pain"


def test_deleet_maps_digit_substitutions() -> None:
    assert normalize.deleet("k1ll my53lf") == "kill myself"


def test_squeeze_defeats_letter_spacing() -> None:
    assert normalize.squeeze("k i l l  m y s e l f") == "killmyself"


def test_detection_forms_deduplicates_when_identical() -> None:
    assert normalize.detection_forms("plain text") == ("plain text",)
    assert len(normalize.detection_forms("k1ll")) == 2


def test_phrase_pattern_tolerates_separators_but_respects_boundaries() -> None:
    import re

    pattern = re.compile(normalize.phrase_pattern("chest pain"), re.IGNORECASE)
    assert pattern.search("chest-pain")
    assert pattern.search("chest   pain")
    assert not pattern.search("chestpainless")

    die = re.compile(normalize.phrase_pattern("die"), re.IGNORECASE)
    assert not die.search("diet")


def test_excerpt_is_single_line_and_bounded() -> None:
    long_text = "a" * 200
    out = normalize.excerpt(long_text, limit=20)
    assert len(out) == 20
    assert "\n" not in normalize.excerpt("two\nlines")
