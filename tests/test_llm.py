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
        vector = self._embeddings.pop(0)
        return SimpleNamespace(data=[SimpleNamespace(embedding=vector)])


def _client(monkeypatch, responses: list[str], max_retries: int = 3) -> tuple[LLMClient, FakeClient]:
    fake = FakeClient(responses=responses)
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake)
    cfg = LLMConfig(chat_model="test-model", max_retries=max_retries)
    return LLMClient(cfg), fake


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
        client.embed("some text")


def test_embed_returns_vector_from_response(monkeypatch):
    fake = FakeClient(embeddings=[[0.1, 0.2, 0.3]])
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake)
    client = LLMClient(LLMConfig(chat_model="test-model", embedding_model="test-embed-model"))
    result = client.embed("some text")
    assert result == [0.1, 0.2, 0.3]
    assert fake.embedding_calls[0]["model"] == "test-embed-model"
    assert fake.embedding_calls[0]["input"] == "some text"


def test_extract_statements_parses_valid_json(monkeypatch):
    client, fake = _client(
        monkeypatch,
        ['{"statements": [{"text": "Works as a nurse", "message_indices": [0, 1]}]}'],
    )
    result = client.extract_statements("transcript text", "Alex")
    assert result == [{"text": "Works as a nurse", "message_indices": [0, 1]}]
    assert len(fake.calls) == 1
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


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
