from chatmem.parsers.text import fix_mojibake, normalize_alias


def _mojibake(s: str) -> str:
    """Mangle a proper UTF-8 string the way Instagram's export does."""
    return s.encode("utf-8").decode("latin-1")


def test_fix_mojibake_round_trips_emoji():
    assert fix_mojibake(_mojibake("\U0001f914")) == "\U0001f914"  # thinking face
    assert fix_mojibake(_mojibake("\U0001f62d")) == "\U0001f62d"  # loudly crying
    assert fix_mojibake(_mojibake("\U0001f602")) == "\U0001f602"  # tears of joy


def test_fix_mojibake_round_trips_multi_codepoint_text():
    original = "Easy for you to do \U0001f624\U0001f624\nI can't be cutting worms at home \U0001f62d"
    assert fix_mojibake(_mojibake(original)) == original


def test_fix_mojibake_leaves_plain_ascii_untouched():
    assert fix_mojibake("just plain ascii text") == "just plain ascii text"


def test_fix_mojibake_leaves_already_correct_non_latin_text_untouched():
    devanagari = "नमस्ते, कैसे हो?"
    assert fix_mojibake(devanagari) == devanagari


def test_fix_mojibake_leaves_bare_emoji_untouched():
    # A correctly-decoded emoji does not round-trip through latin-1 (it will
    # raise UnicodeEncodeError), so it must be returned unchanged rather
    # than mangled further.
    assert fix_mojibake("\U0001f600") == "\U0001f600"


def test_normalize_alias_case_and_whitespace():
    assert normalize_alias("Alex   Rivera") == normalize_alias("alex rivera")
    assert normalize_alias(" Alex Rivera ") == normalize_alias("Alex Rivera")


def test_normalize_alias_fixes_mojibake_first():
    mangled = _mojibake("Alex \U0001f600 Rivera")
    assert normalize_alias(mangled) == normalize_alias("Alex \U0001f600 Rivera")


def test_normalize_alias_distinguishes_different_names():
    assert normalize_alias("Alex Rivera") != normalize_alias("Alex R.")
