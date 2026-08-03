"""Extraction pipeline: turn one session's messages into stored self-statements.

Per-session flow: build a numbered transcript, ask the LLM to extract the
target's self-statements with citations, optionally re-check each one against
the transcript (extraction.validation_pass), then map citations back to
message ids/timestamps for storage.
"""

from __future__ import annotations

from datetime import datetime, timezone

from chatmem.llm import LLMClient
from chatmem.models import Message, Session, Statement


def _render_message(m: Message) -> str:
    if m.text:
        return m.text
    if m.media_type:
        return f"[shared {m.media_type}]"
    return "[empty message]"


def build_transcript(messages: list[Message], display_name_by_person: dict[str, str]) -> str:
    """Render messages as a numbered transcript.

    Index i in the rendered text corresponds to messages[i] -- callers map a
    model's message_indices back through that same list.
    """
    lines = [
        f"[{i}] {display_name_by_person.get(m.person_id, m.sender)}: {_render_message(m)}"
        for i, m in enumerate(messages)
    ]
    return "\n".join(lines)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def extract_session(
    session: Session,
    messages: list[Message],
    target_person_id: str,
    target_name: str,
    llm: LLMClient,
    display_name_by_person: dict[str, str],
    validation_pass: bool = False,
) -> list[Statement]:
    """Extract the target's self-statements from one session.

    Returns an empty list without calling the LLM if the target didn't
    participate in this session.
    """
    if target_person_id not in {m.person_id for m in messages}:
        return []

    transcript = build_transcript(messages, display_name_by_person)
    raw_statements = llm.extract_statements(transcript, target_name)

    candidates: list[tuple[str, list[Message]]] = []
    for raw in raw_statements:
        text = raw.get("text")
        if not text:
            continue
        indices = raw.get("message_indices") or []
        cited = [messages[i] for i in indices if isinstance(i, int) and 0 <= i < len(messages)]
        candidates.append((text, cited))

    if validation_pass and candidates:
        supported = llm.validate_statements(
            transcript, target_name, [text for text, _ in candidates]
        )
        candidates = [c for c, ok in zip(candidates, supported) if ok]

    created_at = _now_iso()
    statements: list[Statement] = []
    for text, cited in candidates:
        if cited:
            source_ids = [m.id for m in cited]
            start_ts = min(m.timestamp_utc for m in cited)
            end_ts = max(m.timestamp_utc for m in cited)
        else:
            source_ids = []
            start_ts = session.start_ts
            end_ts = session.end_ts
        statements.append(
            Statement(
                id=None,
                person_id=target_person_id,
                session_id=session.id,
                thread_id=session.thread_id,
                text=text,
                source_message_ids=source_ids,
                start_ts=start_ts,
                end_ts=end_ts,
                created_at=created_at,
            )
        )
    return statements
