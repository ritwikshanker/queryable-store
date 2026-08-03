"""Prompt for the extraction pass: turn a session transcript into self-statements."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You read a chat session transcript and extract statements a specific person made \
about themself. Every message is numbered.

Rules:
- Only extract statements from the target person's own messages. Never extract a \
statement from what someone else said, even if it is about the target.
- Only extract statements that assert a fact, preference, opinion, or event about the \
target -- not questions, greetings, small talk, or reactions with no content.
- Do not infer or guess anything the transcript does not directly support. If nothing \
qualifies, return an empty list.
- Each statement must be self-contained: a reader with no other context should \
understand it (e.g. "Works as a nurse at a hospital in Chicago", not "Works there too").
- For each statement, cite every message index that supports it.

Respond with JSON only, matching this shape exactly:
{"statements": [{"text": "...", "message_indices": [0, 2]}]}
"""


def build_user_message(transcript: str, target_name: str) -> str:
    return (
        f"Target person: {target_name}\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Extract {target_name}'s self-statements as JSON."
    )
