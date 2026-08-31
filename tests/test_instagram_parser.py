import json
from pathlib import Path

import pytest

from chatmem.parsers.instagram import InstagramParser


def _write(thread_dir: Path, filename: str, participants: list[str], messages: list[dict]) -> None:
    thread_dir.mkdir(parents=True, exist_ok=True)
    (thread_dir / filename).write_text(
        json.dumps({"participants": [{"name": p} for p in participants], "messages": messages}),
        encoding="utf-8",
    )


def _base_msg(sender: str, ts: int, **extra) -> dict:
    d = {"sender_name": sender, "timestamp_ms": ts, "is_geoblocked_for_viewer": False}
    d.update(extra)
    return d


def test_detect_single_thread_dir(tmp_path):
    thread_dir = tmp_path / "thread"
    _write(thread_dir, "message_1.json", ["A", "B"], [_base_msg("A", 1, content="hi")])
    assert InstagramParser().detect(thread_dir) is True


def test_detect_inbox_root(tmp_path):
    inbox = tmp_path / "inbox"
    thread_dir = inbox / "thread"
    _write(thread_dir, "message_1.json", ["A", "B"], [_base_msg("A", 1, content="hi")])
    assert InstagramParser().detect(inbox) is True


def test_detect_false_for_unrelated_dir(tmp_path):
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    assert InstagramParser().detect(empty) is False


def test_merges_and_sorts_ascending_across_split_files(tmp_path):
    thread_dir = tmp_path / "thread"
    # message_2.json = older half, message_1.json = newer half, both
    # stored newest-first within the file (as the real export does).
    older = [_base_msg("A", 300, content="c"), _base_msg("B", 200, content="b"), _base_msg("A", 100, content="a")]
    newer = [_base_msg("B", 600, content="f"), _base_msg("A", 500, content="e"), _base_msg("B", 400, content="d")]
    _write(thread_dir, "message_2.json", ["A", "B"], older)
    _write(thread_dir, "message_1.json", ["A", "B"], newer)

    thread = next(InstagramParser().parse(thread_dir))

    assert [m.text for m in thread.messages] == ["a", "b", "c", "d", "e", "f"]
    assert [m.timestamp_ms for m in thread.messages] == [100, 200, 300, 400, 500, 600]


def test_tied_timestamps_are_stable_and_deterministic(tmp_path):
    thread_dir = tmp_path / "thread"
    msgs = [
        _base_msg("A", 100, content="third"),
        _base_msg("B", 100, content="second"),
        _base_msg("A", 100, content="first"),
    ]
    _write(thread_dir, "message_1.json", ["A", "B"], msgs)

    first_pass = [m.text for m in next(InstagramParser().parse(thread_dir)).messages]
    second_pass = [m.text for m in next(InstagramParser().parse(thread_dir)).messages]

    assert first_pass == second_pass
    assert set(first_pass) == {"first", "second", "third"}


def test_drops_unsent(tmp_path):
    thread_dir = tmp_path / "thread"
    _write(thread_dir, "message_1.json", ["A", "B"], [_base_msg("A", 1, content="oops", is_unsent=True)])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.messages == []
    assert thread.dropped == {"unsent": 1}


def test_drops_call_via_call_duration_field(tmp_path):
    thread_dir = tmp_path / "thread"
    _write(thread_dir, "message_1.json", ["A", "B"], [_base_msg("A", 1, call_duration=42)])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.dropped == {"call": 1}


def test_drops_call_via_content_pattern(tmp_path):
    thread_dir = tmp_path / "thread"
    _write(thread_dir, "message_1.json", ["A", "B"], [_base_msg("A", 1, content="Missed video chat")])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.dropped == {"call": 1}


def test_drops_liked_a_message(tmp_path):
    thread_dir = tmp_path / "thread"
    _write(thread_dir, "message_1.json", ["A", "B"], [_base_msg("A", 1, content="Liked a message")])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.dropped == {"reaction": 1}


def test_drops_reacted_to_your_message(tmp_path):
    thread_dir = tmp_path / "thread"
    _write(thread_dir, "message_1.json", ["A", "B"], [_base_msg("A", 1, content="Reacted \U0001f602 to your message")])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.dropped == {"reaction": 1}


def test_drops_empty_message_with_no_content_and_no_media(tmp_path):
    thread_dir = tmp_path / "thread"
    # No "content" key at all -- exactly the shape a real export can produce.
    raw = {"sender_name": "A", "timestamp_ms": 1, "is_geobloced_for_viewer": False}
    _write(thread_dir, "message_1.json", ["A", "B"], [raw])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.messages == []
    assert thread.dropped == {"empty": 1}


def test_keeps_message_with_reactions_array(tmp_path):
    thread_dir = tmp_path / "thread"
    raw = _base_msg("A", 1, content="real message", reactions=[{"reaction": "\U0001f602", "actor": "B"}])
    _write(thread_dir, "message_1.json", ["A", "B"], [raw])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.dropped == {}
    assert len(thread.messages) == 1
    assert thread.messages[0].text == "real message"


@pytest.mark.parametrize(
    "field,value,expected_media_type",
    [
        ("photos", [{"uri": "p.jpg"}], "photo"),
        ("videos", [{"uri": "v.mp4"}], "video"),
        ("audio_files", [{"uri": "a.m4a"}], "audio"),
        ("gifs", [{"uri": "g.gif"}], "gif"),
        ("files", [{"uri": "f.pdf"}], "file"),
        ("sticker", {"uri": "s.webp"}, "sticker"),
    ],
)
def test_media_placeholder_kept_with_no_text(tmp_path, field, value, expected_media_type):
    thread_dir = tmp_path / "thread"
    raw = _base_msg("A", 1, **{field: value})
    _write(thread_dir, "message_1.json", ["A", "B"], [raw])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.dropped == {}
    assert len(thread.messages) == 1
    assert thread.messages[0].media_type == expected_media_type
    assert thread.messages[0].text is None


def test_share_media_uses_share_text(tmp_path):
    thread_dir = tmp_path / "thread"
    raw = _base_msg("A", 1, share={"link": "https://x", "share_text": "look at this"})
    _write(thread_dir, "message_1.json", ["A", "B"], [raw])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.messages[0].media_type == "share"
    assert thread.messages[0].text == "look at this"


def test_share_media_with_no_share_text_has_null_text(tmp_path):
    thread_dir = tmp_path / "thread"
    raw = _base_msg("A", 1, share={"link": "https://x"})
    _write(thread_dir, "message_1.json", ["A", "B"], [raw])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.messages[0].media_type == "share"
    assert thread.messages[0].text is None


def test_mojibake_fixed_on_content_and_sender(tmp_path):
    thread_dir = tmp_path / "thread"
    mangled_content = "hi \U0001f600".encode("utf-8").decode("latin-1")
    mangled_sender = "R\U000000e9mi".encode("utf-8").decode("latin-1")
    raw = _base_msg(mangled_sender, 1, content=mangled_content)
    _write(thread_dir, "message_1.json", [mangled_sender, "B"], [raw])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.messages[0].text == "hi \U0001f600"
    assert thread.messages[0].sender == "Rémi"


def test_typo_field_names_do_not_crash(tmp_path):
    thread_dir = tmp_path / "thread"
    raw = {
        "sender_name": "A",
        "timestamp_ms": 1,
        "content": "hello",
        "is_geobloced_for_viewer": False,  # typo, as seen in a real export
        "is_unsent_image_by_messenger_kid_parent": False,
    }
    _write(thread_dir, "message_1.json", ["A", "B"], [raw])
    thread = next(InstagramParser().parse(thread_dir))
    assert thread.messages[0].text == "hello"


def test_invalid_json_raises_with_file_and_offset(tmp_path):
    thread_dir = tmp_path / "thread"
    thread_dir.mkdir(parents=True)
    (thread_dir / "message_1.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        list(InstagramParser().parse(thread_dir))

    msg = str(exc_info.value)
    assert "message_1.json" in msg
    assert "line" in msg and "column" in msg


def test_synthetic_archive_fixture(synthetic_archive):
    parser = InstagramParser()
    threads = {t.thread_id: t for t in parser.parse(synthetic_archive)}

    assert set(threads) == {"thread_alpha", "thread_beta"}

    alpha = threads["thread_alpha"]
    assert alpha.dropped == {"empty": 1, "reaction": 1, "call": 1, "unsent": 1}
    assert len(alpha.messages) == 23
    assert set(alpha.participants) == {"Alex Rivera", "Sam Chen"}
    # ascending order maintained across the two split files
    timestamps = [m.timestamp_ms for m in alpha.messages]
    assert timestamps == sorted(timestamps)

    beta = threads["thread_beta"]
    assert beta.dropped == {}
    assert len(beta.messages) == 5
    assert set(beta.participants) == {"Alex R.", "Sam Chen"}


def test_messages_without_usable_timestamps_are_dropped_not_dated_to_1970(tmp_path):
    thread_dir = tmp_path / "thread"
    _write(
        thread_dir,
        "message_1.json",
        ["A", "B"],
        [
            _base_msg("A", 3000, content="real"),
            {"sender_name": "A", "content": "no timestamp at all"},
            {"sender_name": "A", "timestamp_ms": "oops", "content": "bad type"},
            {"sender_name": "A", "timestamp_ms": 0, "content": "epoch zero"},
        ],
    )
    [thread] = list(InstagramParser().parse(thread_dir))

    # Kept messages are only the ones that can be placed in time; the rest
    # are reported rather than silently sorted to 1970.
    assert [m.text for m in thread.messages] == ["real"]
    assert thread.dropped["missing_timestamp"] == 3


def test_reaction_and_call_rows_with_surrounding_whitespace_are_dropped(tmp_path):
    """Real exports emit "Reacted <emoji> to your message " with a trailing
    space. An anchored pattern that is not whitespace-tolerant lets thousands
    of these through as if they were real messages -- 994 of them in a single
    real archive."""
    thread = tmp_path / "t1"
    _write(
        thread,
        "message_1.json",
        ["A", "B"],
        [
            {"sender_name": "A", "timestamp_ms": 4, "content": "Reacted \u00f0\u009f\u0098\u0082 to your message "},
            {"sender_name": "A", "timestamp_ms": 3, "content": "Liked a message  "},
            {"sender_name": "B", "timestamp_ms": 2, "content": "  Missed a video call "},
            {"sender_name": "B", "timestamp_ms": 1, "content": "a real message"},
        ],
    )
    parsed = list(InstagramParser().parse(thread))[0]
    assert [m.text for m in parsed.messages] == ["a real message"]
    assert parsed.dropped["reaction"] == 2
    assert parsed.dropped["call"] == 1
