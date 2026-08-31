from datetime import datetime, timezone

from chatmem.parsers.whatsapp import WhatsAppParser, _slug

ANDROID = """\
12/05/2023, 9:41 AM - Messages and calls are end-to-end encrypted.
12/05/2023, 9:41 AM - Dana: hey there
12/05/2023, 9:42 AM - Alex: hi! I moved to Berlin last month
12/05/2023, 9:43 AM - Dana: <Media omitted>
"""

# iOS layout, complete with the invisible marks WhatsApp inserts.
IOS = (
    "‎[12/05/2023, 9:41:02 AM] Dana: hey there\n"
    "[12/05/2023, 9:42:10 AM] Alex: hi!\n"
)


def _write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def _ts(y, m, d, hh, mm, ss=0):
    return int(datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc).timestamp() * 1000)


def test_detect_android_chat_file(tmp_path):
    path = _write(tmp_path, "WhatsApp Chat with Dana.txt", ANDROID)
    assert WhatsAppParser().detect(path) is True


def test_detect_directory_containing_chat_export(tmp_path):
    _write(tmp_path, "_chat.txt", IOS)
    assert WhatsAppParser().detect(tmp_path) is True


def test_detect_false_for_unrelated_text_file(tmp_path):
    path = _write(tmp_path, "notes.txt", "just some notes\nnothing timestamped here\n")
    assert WhatsAppParser().detect(path) is False


def test_parses_android_export(tmp_path):
    path = _write(tmp_path, "WhatsApp Chat with Dana.txt", ANDROID)
    [thread] = list(WhatsAppParser().parse(path))

    assert thread.source == "whatsapp"
    assert thread.thread_id == "whatsapp_chat_with_dana"
    assert set(thread.participants) == {"Dana", "Alex"}
    assert [m.text for m in thread.messages] == [
        "hey there",
        "hi! I moved to Berlin last month",
        None,  # the media placeholder
    ]
    assert thread.messages[2].media_type == "media"
    assert thread.dropped["system"] == 1  # the encryption notice
    assert thread.messages[0].timestamp_ms == _ts(2023, 5, 12, 9, 41)


def test_parses_ios_export_with_invisible_characters(tmp_path):
    path = _write(tmp_path, "_chat.txt", IOS)
    [thread] = list(WhatsAppParser().parse(path))

    assert [m.sender for m in thread.messages] == ["Dana", "Alex"]
    assert [m.text for m in thread.messages] == ["hey there", "hi!"]
    assert thread.messages[0].timestamp_ms == _ts(2023, 5, 12, 9, 41, 2)


def test_multiline_message_is_joined_onto_the_previous_one(tmp_path):
    body = (
        "12/05/2023, 9:41 AM - Dana: first line\n"
        "second line\n"
        "third line\n"
        "12/05/2023, 9:42 AM - Alex: separate message\n"
    )
    path = _write(tmp_path, "chat.txt", body)
    [thread] = list(WhatsAppParser().parse(path))

    assert thread.messages[0].text == "first line\nsecond line\nthird line"
    assert thread.messages[1].text == "separate message"


def test_media_placeholders_map_to_media_types(tmp_path):
    body = (
        "12/05/2023, 9:41 AM - Dana: image omitted\n"
        "12/05/2023, 9:42 AM - Dana: video omitted\n"
        "12/05/2023, 9:43 AM - Dana: audio omitted\n"
        "12/05/2023, 9:44 AM - Dana: sticker omitted\n"
    )
    path = _write(tmp_path, "chat.txt", body)
    [thread] = list(WhatsAppParser().parse(path))

    assert [m.media_type for m in thread.messages] == ["photo", "video", "audio", "sticker"]
    assert all(m.text is None for m in thread.messages)


def test_deleted_and_system_messages_are_counted_not_emitted(tmp_path):
    body = (
        "12/05/2023, 9:41 AM - Messages and calls are end-to-end encrypted.\n"
        "12/05/2023, 9:42 AM - Dana: This message was deleted\n"
        "12/05/2023, 9:43 AM - Dana: real message\n"
        "12/05/2023, 9:44 AM - Alex created this group\n"
    )
    path = _write(tmp_path, "chat.txt", body)
    [thread] = list(WhatsAppParser().parse(path))

    assert [m.text for m in thread.messages] == ["real message"]
    assert thread.dropped["deleted"] == 1
    assert thread.dropped["system"] == 2


def test_day_first_inferred_from_a_component_above_twelve(tmp_path):
    # 25 can only be a day, so the whole file is read as DD/MM.
    body = (
        "25/05/2023, 9:41 AM - Dana: later\n"
        "03/06/2023, 9:41 AM - Dana: even later\n"
    )
    path = _write(tmp_path, "chat.txt", body)
    [thread] = list(WhatsAppParser().parse(path))

    assert thread.messages[0].timestamp_ms == _ts(2023, 5, 25, 9, 41)
    assert thread.messages[1].timestamp_ms == _ts(2023, 6, 3, 9, 41)


def test_month_first_inferred_from_a_component_above_twelve(tmp_path):
    body = (
        "05/25/2023, 9:41 AM - Dana: later\n"
        "06/03/2023, 9:41 AM - Dana: even later\n"
    )
    path = _write(tmp_path, "chat.txt", body)
    [thread] = list(WhatsAppParser().parse(path))

    assert thread.messages[0].timestamp_ms == _ts(2023, 5, 25, 9, 41)


def test_ambiguous_dates_resolved_by_chronological_order(tmp_path):
    # Both components are <= 12. Read day-first the dates run backwards, so
    # month-first is the only chronological reading.
    body = (
        "05/06/2023, 9:41 AM - Dana: first\n"
        "06/03/2023, 9:41 AM - Dana: second\n"
    )
    path = _write(tmp_path, "chat.txt", body)
    [thread] = list(WhatsAppParser().parse(path))

    assert thread.messages[0].timestamp_ms == _ts(2023, 5, 6, 9, 41)
    assert thread.messages[1].timestamp_ms == _ts(2023, 6, 3, 9, 41)


def test_fully_ambiguous_dates_default_to_day_first(tmp_path):
    # Chronological under either reading, so the documented default applies.
    body = (
        "01/02/2023, 9:41 AM - Dana: first\n"
        "02/03/2023, 9:41 AM - Dana: second\n"
    )
    path = _write(tmp_path, "chat.txt", body)
    [thread] = list(WhatsAppParser().parse(path))

    assert thread.messages[0].timestamp_ms == _ts(2023, 2, 1, 9, 41)


def test_24_hour_times_are_supported(tmp_path):
    body = "12/05/2023, 21:41 - Dana: evening\n"
    path = _write(tmp_path, "chat.txt", body)
    [thread] = list(WhatsAppParser().parse(path))
    assert thread.messages[0].timestamp_ms == _ts(2023, 5, 12, 21, 41)


def test_slug_keeps_non_latin_chat_names_distinct():
    """A name in a non-Latin script reduces to the boilerplate around it --
    "WhatsApp Chat with <Devanagari>" collapses to "whatsapp_chat_with" -- so
    every such export would land on one thread id and silently replace the
    previous thread on ingest."""
    a = _slug("WhatsApp Chat with \u0905\u0923\u093f\u092e\u093e\u0902")
    b = _slug("WhatsApp Chat with \uc544\ub2c8\ub9c8")
    assert a != b
    assert a.startswith("whatsapp_chat_with_")
    assert a == _slug("WhatsApp Chat with \u0905\u0923\u093f\u092e\u093e\u0902")
    assert a.isascii(), "ids get typed at a shell for --thread"


def test_slug_leaves_ascii_names_readable():
    """The digest is only for names that would otherwise be lost; an ASCII
    name must keep the plain, guessable id it already had."""
    assert _slug("WhatsApp Chat with Alex") == "whatsapp_chat_with_alex"
    # Separators are noise in both forms, so dropping them is not information
    # loss and must not trigger a digest.
    assert _slug("_Chat_") == "chat"
    assert _slug("") == "whatsapp_thread"
