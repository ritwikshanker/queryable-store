"""Prompt for `chatmem query --answer`: compose an answer from retrieved statements.

The statements are the only evidence the model gets. Everything it says has to
trace back to one of them, and the citation markers line up with the numbered
list the CLI prints underneath the answer.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You answer questions about a person using only a numbered list of statements \
previously extracted from their own chat messages. Each statement is evidence; \
nothing else is.

Rules:
- Use only the statements provided. Do not add facts from your own knowledge.
- Cite the statements you rely on with their numbers in square brackets, like [1] \
or [1][3], immediately after the claim they support.
- If the statements do not answer the question, say so plainly instead of guessing.
- If the statements disagree with each other, say that and cite both.
- Answer in a few sentences of plain prose. Do not repeat the list back.
"""


def build_user_message(question: str, statements: list[str]) -> str:
    numbered = "\n".join(f"[{i + 1}] {text}" for i, text in enumerate(statements))
    return (
        f"Question: {question}\n\n"
        f"Statements:\n{numbered}\n\n"
        "Answer the question using only these statements, citing them by number."
    )
