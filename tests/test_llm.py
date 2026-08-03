from types import SimpleNamespace

import pytest

from chatmem import llm as llm_module
from chatmem.config import LLMConfig
from chatmem.llm import LLMClient, LLMConfigError, LLMResponseError


class FakeClient:
    """Stands in for openai.OpenAI(): records calls, replays scripted responses."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self._responses.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _client(monkeypatch, responses: list[str], max_retries: int = 3) -> tuple[LLMClient, FakeClient]:
    fake = FakeClient(responses)
    monkeypatch.setattr(llm_module, "OpenAI", lambda **kwargs: fake)
    cfg = LLMConfig(chat_model="test-model", max_retries=max_retries)
    return LLMClient(cfg), fake


def test_missing_chat_model_raises_config_error():
    with pytest.raises(LLMConfigError):
        LLMClient(LLMConfig(chat_model=""))


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
