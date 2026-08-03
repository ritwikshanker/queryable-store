"""Prompt for the validation pass: re-check extracted statements against the transcript."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You check whether previously extracted statements about a target person are actually \
supported by their own words in a chat session transcript. Every message is numbered, \
and every candidate statement is numbered.

A statement is supported only if the target person's own messages in the transcript \
assert it. A statement is not supported if it relies on inference beyond what the \
target actually said, or if it is only supported by someone else's message.

Respond with JSON only, matching this shape exactly, with one entry per candidate \
statement index:
{"results": [{"index": 0, "supported": true}]}
"""


def build_user_message(transcript: str, target_name: str, statements: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(statements))
    return (
        f"Target person: {target_name}\n\n"
        f"Transcript:\n{transcript}\n\n"
        f"Candidate statements:\n{numbered}\n\n"
        "Check each candidate statement against the transcript and respond as JSON."
    )
