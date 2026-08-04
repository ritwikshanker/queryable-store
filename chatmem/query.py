"""Semantic query: rank stored statements by cosine similarity to a question.

Plain Python, no numpy -- personal-scale statement counts don't need it.
"""

from __future__ import annotations

import math

from chatmem.llm import LLMClient
from chatmem.models import Statement


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_statements(
    query_embedding: list[float],
    statements: list[Statement],
    limit: int,
    min_score: float | None = None,
) -> list[tuple[Statement, float]]:
    """Rank by similarity, then cut. min_score is applied before the limit, so
    a weak match never fills a slot just because nothing better exists."""
    scored = [
        (s, cosine_similarity(query_embedding, s.embedding))
        for s in statements
        if s.embedding is not None
    ]
    if min_score is not None:
        scored = [pair for pair in scored if pair[1] >= min_score]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]


def query(
    question: str,
    statements: list[Statement],
    llm: LLMClient,
    limit: int = 5,
    min_score: float | None = None,
) -> list[tuple[Statement, float]]:
    [query_embedding] = llm.embed([question])
    return rank_statements(query_embedding, statements, limit, min_score=min_score)
