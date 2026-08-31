"""Prompt for the classification pass: file each statement under one topic.

Statements are sent in batches with their position in the batch as the id, so
the model never sees a database id and a misnumbered response can only ever
corrupt the batch it came from.
"""

from __future__ import annotations

from chatmem import topics

SYSTEM_PROMPT = f"""\
You file short factual statements about one person under exactly one topic each.

Topics:
{topics.catalog()}

Rules:
- Choose exactly one topic per statement, using its key exactly as written above.
- Judge only the statement's own words. Do not infer beyond them.
- Pick the most specific topic that fits. Use "other" only when nothing else does.
- Return a result for every statement you were given, keyed by the index it was \
given under.

Respond with JSON only, matching this shape exactly:
{{"assignments": [{{"index": 0, "topic": "family"}}]}}
"""


def build_user_message(statements: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {text}" for i, text in enumerate(statements))
    return f"Statements:\n{numbered}\n\nAssign a topic to each of the {len(statements)} statements."
