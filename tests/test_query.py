from chatmem.models import Statement
from chatmem.query import cosine_similarity, query, rank_statements


def _statement(text, embedding, session_id=1):
    return Statement(
        id=None,
        person_id="target",
        session_id=session_id,
        thread_id="t1",
        text=text,
        source_message_ids=[],
        start_ts="1970-01-01T00:00:00.000000Z",
        end_ts="1970-01-01T00:00:01.000000Z",
        created_at="1970-01-01T00:00:02.000000Z",
        embedding=embedding,
    )


class FakeLLM:
    def __init__(self, embedding):
        self._embedding = embedding

    def embed(self, text):
        return self._embedding


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_rank_statements_orders_by_similarity_descending():
    close = _statement("close match", [1.0, 0.0])
    far = _statement("far match", [0.0, 1.0])
    results = rank_statements([1.0, 0.0], [far, close], limit=10)
    assert [s.text for s, _ in results] == ["close match", "far match"]
    assert results[0][1] == 1.0
    assert results[1][1] == 0.0


def test_rank_statements_respects_limit():
    statements = [_statement(f"s{i}", [1.0, 0.0]) for i in range(5)]
    results = rank_statements([1.0, 0.0], statements, limit=2)
    assert len(results) == 2


def test_rank_statements_skips_unembedded_statements():
    embedded = _statement("has embedding", [1.0, 0.0])
    unembedded = _statement("no embedding", None)
    results = rank_statements([1.0, 0.0], [embedded, unembedded], limit=10)
    assert [s.text for s, _ in results] == ["has embedding"]


def test_query_embeds_question_and_ranks_statements():
    match = _statement("relevant", [1.0, 0.0])
    other = _statement("irrelevant", [0.0, 1.0])
    llm = FakeLLM(embedding=[1.0, 0.0])
    results = query("some question", [match, other], llm, limit=1)
    assert [s.text for s, _ in results] == ["relevant"]
