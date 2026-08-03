from chatmem.extract import build_transcript, extract_session
from chatmem.models import Message, Session


def _msg(id_, sender, person_id, seq, ts_ms, text="hi", media_type=None):
    return Message(
        id=id_,
        thread_id="t1",
        sender=sender,
        person_id=person_id,
        timestamp_utc=f"1970-01-01T00:00:{ts_ms:02d}.000000Z",
        timestamp_ms=ts_ms,
        text=text,
        media_type=media_type,
        seq=seq,
    )


def _session(start_seq=0, end_seq=2):
    return Session(
        id=1,
        thread_id="t1",
        start_seq=start_seq,
        end_seq=end_seq,
        start_ts="1970-01-01T00:00:00.000000Z",
        end_ts="1970-01-01T00:00:02.000000Z",
        message_count=end_seq - start_seq + 1,
    )


NAMES = {"target": "Alex Rivera", "other": "Sam"}


class FakeLLM:
    def __init__(self, statements, supported=None):
        self._statements = statements
        self._supported = supported

    def extract_statements(self, transcript, target_name):
        return self._statements

    def validate_statements(self, transcript, target_name, statements):
        return self._supported

    def embed(self, text):
        # Deterministic and distinct per input, just enough to assert on.
        return [float(len(text))]


def test_build_transcript_numbers_lines_and_labels_media():
    messages = [
        _msg("m0", "Alex Rivera", "target", 0, 0, text="hi there"),
        _msg("m1", "Sam", "other", 1, 1, text=None, media_type="photo"),
    ]
    transcript = build_transcript(messages, NAMES)
    assert transcript == "[0] Alex Rivera: hi there\n[1] Sam: [shared photo]"


def test_extract_session_skips_when_target_absent():
    messages = [_msg("m0", "Sam", "other", 0, 0)]
    llm = FakeLLM(statements=[{"text": "should not be reached", "message_indices": [0]}])
    result = extract_session(_session(0, 0), messages, "target", "Alex Rivera", llm, NAMES)
    assert result == []


def test_extract_session_maps_citations_to_messages():
    messages = [
        _msg("m0", "Alex Rivera", "target", 0, 10, text="I work as a nurse"),
        _msg("m1", "Sam", "other", 1, 11, text="oh nice"),
        _msg("m2", "Alex Rivera", "target", 2, 12, text="in Chicago"),
    ]
    llm = FakeLLM(
        statements=[{"text": "Works as a nurse in Chicago", "message_indices": [0, 2]}]
    )
    result = extract_session(_session(0, 2), messages, "target", "Alex Rivera", llm, NAMES)
    assert len(result) == 1
    s = result[0]
    assert s.text == "Works as a nurse in Chicago"
    assert s.source_message_ids == ["m0", "m2"]
    assert s.start_ts == messages[0].timestamp_utc
    assert s.end_ts == messages[2].timestamp_utc
    assert s.person_id == "target"
    assert s.session_id == 1
    assert s.thread_id == "t1"
    assert s.embedding == [float(len("Works as a nurse in Chicago"))]


def test_extract_session_falls_back_to_session_bounds_when_no_citation():
    messages = [_msg("m0", "Alex Rivera", "target", 0, 0)]
    llm = FakeLLM(statements=[{"text": "no citation given", "message_indices": []}])
    session = _session(0, 0)
    result = extract_session(session, messages, "target", "Alex Rivera", llm, NAMES)
    assert result[0].source_message_ids == []
    assert result[0].start_ts == session.start_ts
    assert result[0].end_ts == session.end_ts


def test_extract_session_drops_statements_missing_text():
    messages = [_msg("m0", "Alex Rivera", "target", 0, 0)]
    llm = FakeLLM(statements=[{"text": "", "message_indices": [0]}, {"message_indices": [0]}])
    result = extract_session(_session(0, 0), messages, "target", "Alex Rivera", llm, NAMES)
    assert result == []


def test_extract_session_validation_pass_drops_unsupported():
    messages = [_msg("m0", "Alex Rivera", "target", 0, 0, text="I like hiking")]
    llm = FakeLLM(
        statements=[
            {"text": "Likes hiking", "message_indices": [0]},
            {"text": "Made up fact", "message_indices": [0]},
        ],
        supported=[True, False],
    )
    result = extract_session(
        _session(0, 0), messages, "target", "Alex Rivera", llm, NAMES, validation_pass=True
    )
    assert [s.text for s in result] == ["Likes hiking"]


def test_extract_session_skips_validation_call_when_disabled():
    messages = [_msg("m0", "Alex Rivera", "target", 0, 0)]
    llm = FakeLLM(statements=[{"text": "Likes hiking", "message_indices": [0]}])
    # supported=None on FakeLLM would break if validate_statements were called.
    result = extract_session(
        _session(0, 0), messages, "target", "Alex Rivera", llm, NAMES, validation_pass=False
    )
    assert [s.text for s in result] == ["Likes hiking"]
