from types import SimpleNamespace

import pytest

from chatmem import llm as llm_module
from chatmem.config import LLMConfig
from chatmem.llm import LLMClient, LLMConfigError, LLMResponseError


class FakeClient:
    """Stands in for openai.OpenAI(): records calls, replays scripted responses."""

    def __init__(self, responses: list[str] | None = None, embeddings: list[list[float]] | None = None):
        self._responses = list(responses or [])
        self._embeddings = list(embeddings or [])
        self.calls: list[dict] = []
        self.embedding_calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def _embed(self, **kwargs):
        self.embedding_calls.append(kwargs)
        n = len(kwargs["input"])
        vectors = [self._embeddings.pop(0) for _ in range(n)]
        # Deliberately shuffled: the API doesn't promise response order, so
        # LLMClient.embed must re-order by each datum's index.
        data = [SimpleNamespace(index=i, embedding=v) for i, v in enumerate(vectors)]
        return SimpleNamespace(data=list(reversed(data)))


def _client(monkeypatch, responses: list[str], max_retries: int = 3) -> tuple[LLMClient, FakeClient]:
    fake = FakeClient(responses=responses)
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake)
    cfg = LLMConfig(chat_model="test-model", max_retries=max_retries)
    return LLMClient(cfg), fake


def test_classify_statements_returns_one_topic_per_statement_in_input_order(monkeypatch):
    client, fake = _client(
        monkeypatch,
        ['{"assignments": [{"index": 1, "topic": "work"}, {"index": 0, "topic": "family"}]}'],
    )
    assert client.classify_statements(["has a sister", "is a nurse"]) == ["family", "work"]


def test_classify_statements_makes_no_call_for_an_empty_batch(monkeypatch):
    client, fake = _client(monkeypatch, [])
    assert client.classify_statements([]) == []
    assert fake.calls == []


def test_classify_statements_falls_back_for_a_statement_the_model_skipped(monkeypatch):
    """A missing assignment must not shift the others or drop the statement --
    it renders under Unclassified and the next digest run retries it."""
    from chatmem import topics

    client, _ = _client(monkeypatch, ['{"assignments": [{"index": 0, "topic": "family"}]}'])
    assert client.classify_statements(["a", "b"]) == ["family", topics.FALLBACK_KEY]


def test_classify_statements_falls_back_for_a_topic_outside_the_taxonomy(monkeypatch):
    from chatmem import topics

    client, _ = _client(
        monkeypatch, ['{"assignments": [{"index": 0, "topic": "not_a_real_topic"}]}']
    )
    assert client.classify_statements(["a"]) == [topics.FALLBACK_KEY]


def test_classify_statements_constrains_the_topic_to_the_taxonomy_in_the_schema(monkeypatch):
    """The enum is what keeps a well-behaved model inside the taxonomy; the
    fallback above is only the backstop for one that isn't."""
    from chatmem import topics

    client, fake = _client(monkeypatch, ['{"assignments": []}'])
    client.classify_statements(["a"])
    schema = fake.calls[0]["response_format"]["json_schema"]["schema"]
    enum = schema["properties"]["assignments"]["items"]["properties"]["topic"]["enum"]
    assert enum == list(topics.TOPIC_KEYS)


def test_classify_statements_rejects_a_non_list_assignments_field(monkeypatch):
    client, _ = _client(monkeypatch, ['{"assignments": "family"}'])
    with pytest.raises(LLMResponseError):
        client.classify_statements(["a"])


def test_construction_does_not_require_any_model_id(monkeypatch):
    """LLMClient is used for chat-only (extract) and embedding-only (query)
    work, so __init__ shouldn't demand both up front."""
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: FakeClient())
    LLMClient(LLMConfig(chat_model="", embedding_model=""))  # must not raise


def test_extract_statements_raises_config_error_when_chat_model_unset(monkeypatch):
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: FakeClient())
    client = LLMClient(LLMConfig(chat_model=""))
    with pytest.raises(LLMConfigError):
        client.extract_statements("t", "Alex")


def test_validate_statements_raises_config_error_when_chat_model_unset(monkeypatch):
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: FakeClient())
    client = LLMClient(LLMConfig(chat_model=""))
    with pytest.raises(LLMConfigError):
        client.validate_statements("t", "Alex", ["a"])


def test_embed_raises_config_error_when_embedding_model_unset(monkeypatch):
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: FakeClient())
    client = LLMClient(LLMConfig(chat_model="test-model", embedding_model=""))
    with pytest.raises(LLMConfigError):
        client.embed(["some text"])


def test_embed_returns_vectors_in_input_order_from_one_call(monkeypatch):
    fake = FakeClient(embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake)
    client = LLMClient(LLMConfig(chat_model="test-model", embedding_model="test-embed-model"))

    result = client.embed(["a", "b", "c"])

    # One round-trip for the whole batch, results realigned to input order.
    assert len(fake.embedding_calls) == 1
    assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    assert fake.embedding_calls[0]["model"] == "test-embed-model"
    assert fake.embedding_calls[0]["input"] == ["a", "b", "c"]


def test_embed_short_circuits_on_empty_input(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake)
    client = LLMClient(LLMConfig(chat_model="test-model", embedding_model="test-embed-model"))
    assert client.embed([]) == []
    assert fake.embedding_calls == []


def test_extract_statements_parses_valid_json(monkeypatch):
    client, fake = _client(
        monkeypatch,
        ['{"statements": [{"text": "Works as a nurse", "message_indices": [0, 1]}]}'],
    )
    result = client.extract_statements("transcript text", "Alex")
    assert result == [{"text": "Works as a nurse", "message_indices": [0, 1]}]
    assert len(fake.calls) == 1
    assert fake.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.calls[0]["response_format"]["json_schema"]["name"] == "extraction_result"


def test_extract_statements_retries_on_bad_json_then_succeeds(monkeypatch):
    client, fake = _client(
        monkeypatch,
        ["not json", "still not json", '{"statements": []}'],
        max_retries=3,
    )
    result = client.extract_statements("t", "Alex")
    assert result == []
    assert len(fake.calls) == 3


def test_extract_statements_raises_after_exhausting_retries(monkeypatch):
    client, fake = _client(monkeypatch, ["nope", "nope"], max_retries=2)
    with pytest.raises(LLMResponseError):
        client.extract_statements("t", "Alex")
    assert len(fake.calls) == 2


def test_validate_statements_maps_supported_flags_by_index(monkeypatch):
    client, _ = _client(
        monkeypatch,
        ['{"results": [{"index": 0, "supported": true}, {"index": 1, "supported": false}]}'],
    )
    result = client.validate_statements("t", "Alex", ["a", "b"])
    assert result == [True, False]


def test_validate_statements_defaults_missing_index_to_unsupported(monkeypatch):
    client, _ = _client(monkeypatch, ['{"results": [{"index": 0, "supported": true}]}'])
    result = client.validate_statements("t", "Alex", ["a", "b"])
    assert result == [True, False]


def test_validate_statements_short_circuits_on_empty_input(monkeypatch):
    fake = FakeClient([])
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake)
    client = LLMClient(LLMConfig(chat_model="test-model"))
    assert client.validate_statements("t", "Alex", []) == []
    assert fake.calls == []


def test_reasoning_effort_is_omitted_when_unset(monkeypatch):
    """A server that doesn't know the parameter rejects the whole request, so
    the default must not send it at all."""
    client, fake = _client(monkeypatch, ['{"statements": []}'])
    client.extract_statements("t", "Sam")
    assert "reasoning_effort" not in fake.calls[0]


def test_reasoning_effort_is_passed_through_when_set(monkeypatch):
    fake = FakeClient(responses=['{"statements": []}', "an answer"])
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake)
    client = LLMClient(LLMConfig(chat_model="test-model", reasoning_effort="none"))
    client.extract_statements("t", "Sam")
    client.synthesize_answer("q", ["s"])
    # Both the JSON path and the free-text answer path, or extraction speeds up
    # while `query --answer` silently keeps burning reasoning tokens.
    assert [c["reasoning_effort"] for c in fake.calls] == ["none", "none"]
