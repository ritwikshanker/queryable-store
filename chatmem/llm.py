"""LLM client: thin wrapper around the OpenAI SDK, pointed at LM Studio's
OpenAI-compatible server, for the extraction and validation prompts.
"""

from __future__ import annotations

import json

from openai import OpenAI, OpenAIError

from chatmem import topics
from chatmem.config import LLMConfig
from chatmem.prompts import answer as answer_prompt
from chatmem.prompts import classify as classify_prompt
from chatmem.prompts import extract as extract_prompt
from chatmem.prompts import validate as validate_prompt


class LLMConfigError(ValueError):
    """Raised when a needed model id (chat_model or embedding_model) is not set."""


class LLMResponseError(RuntimeError):
    """Raised when the model's response isn't valid JSON after all retries, or
    doesn't match the shape the prompt asked for."""


class LLMClient:
    """Not every caller needs every model: `chatmem extract` needs chat_model
    (and embedding_model, to embed what it extracts); `chatmem query` only
    needs embedding_model. So config validation happens per-capability in
    _chat_json/embed, not eagerly in __init__.
    """

    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )

    def _extra_chat_kwargs(self) -> dict:
        """Only sent when configured: a server that doesn't know the parameter
        rejects the whole request, so an unset value must not appear at all."""
        if not self._config.reasoning_effort:
            return {}
        return {"reasoning_effort": self._config.reasoning_effort}

    def _chat_json(
        self, system_prompt: str, user_message: str, *, schema_name: str, schema: dict
    ) -> dict:
        if not self._config.chat_model:
            raise LLMConfigError(
                "llm.chat_model is not set in config.yaml -- set it to a chat model id "
                "served by your LM Studio instance."
            )
        last_error: Exception | None = None
        for _ in range(max(1, self._config.max_retries)):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "schema": schema},
                    },
                    **self._extra_chat_kwargs(),
                )
            except OpenAIError as e:
                # Surfaced as our own error so callers (cli.extract) can keep
                # going on a flaky server without importing the OpenAI SDK.
                raise LLMResponseError(f"chat request failed: {e}") from e
            content = response.choices[0].message.content or ""
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                last_error = e
                continue
        raise LLMResponseError(f"model did not return valid JSON after retries: {last_error}")

    def extract_statements(self, transcript: str, target_name: str) -> list[dict]:
        """Returns a list of {"text": str, "message_indices": [int, ...]} dicts."""
        user_message = extract_prompt.build_user_message(transcript, target_name)
        schema = {
            "type": "object",
            "properties": {
                "statements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "message_indices": {"type": "array", "items": {"type": "integer"}},
                        },
                        "required": ["text", "message_indices"],
                    },
                },
            },
            "required": ["statements"],
        }
        data = self._chat_json(
            extract_prompt.SYSTEM_PROMPT,
            user_message,
            schema_name="extraction_result",
            schema=schema,
        )
        statements = data.get("statements", [])
        if not isinstance(statements, list):
            raise LLMResponseError(f"expected 'statements' to be a list, got {type(statements)!r}")
        return statements

    def validate_statements(
        self, transcript: str, target_name: str, statements: list[str]
    ) -> list[bool]:
        """Returns a supported/not-supported bool per statement, same order as input."""
        if not statements:
            return []
        user_message = validate_prompt.build_user_message(transcript, target_name, statements)
        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "supported": {"type": "boolean"},
                        },
                        "required": ["index", "supported"],
                    },
                },
            },
            "required": ["results"],
        }
        data = self._chat_json(
            validate_prompt.SYSTEM_PROMPT,
            user_message,
            schema_name="validation_result",
            schema=schema,
        )
        results = data.get("results", [])
        supported_by_index = {
            r["index"]: bool(r["supported"])
            for r in results
            if isinstance(r, dict) and "index" in r and "supported" in r
        }
        return [supported_by_index.get(i, False) for i in range(len(statements))]

    def classify_statements(self, statements: list[str]) -> list[str]:
        """File each statement under one topic key, same order as input.

        A statement the model skipped, or filed under a key outside the
        taxonomy, falls back to topics.FALLBACK_KEY rather than failing the
        batch -- classification is presentational, and losing a whole batch of
        real statements to one bad key would be the worse outcome.
        """
        if not statements:
            return []
        schema = {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "topic": {"type": "string", "enum": list(topics.TOPIC_KEYS)},
                        },
                        "required": ["index", "topic"],
                    },
                },
            },
            "required": ["assignments"],
        }
        data = self._chat_json(
            classify_prompt.SYSTEM_PROMPT,
            classify_prompt.build_user_message(statements),
            schema_name="classification_result",
            schema=schema,
        )
        assignments = data.get("assignments", [])
        if not isinstance(assignments, list):
            raise LLMResponseError(
                f"expected 'assignments' to be a list, got {type(assignments)!r}"
            )
        topic_by_index = {
            a["index"]: a["topic"]
            for a in assignments
            if isinstance(a, dict)
            and isinstance(a.get("index"), int)
            and a.get("topic") in topics.BY_KEY
        }
        return [
            topic_by_index.get(i, topics.FALLBACK_KEY) for i in range(len(statements))
        ]

    def synthesize_answer(self, question: str, statements: list[str]) -> str:
        """Compose a prose answer from retrieved statements.

        Free text rather than JSON: the answer itself is the product, and the
        [n] citation markers refer to the same numbering the CLI prints.
        """
        if not self._config.chat_model:
            raise LLMConfigError(
                "llm.chat_model is not set in config.yaml -- required to synthesize an answer."
            )
        try:
            response = self._client.chat.completions.create(
                model=self._config.chat_model,
                messages=[
                    {"role": "system", "content": answer_prompt.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": answer_prompt.build_user_message(question, statements),
                    },
                ],
                **self._extra_chat_kwargs(),
            )
        except OpenAIError as e:
            raise LLMResponseError(f"chat request failed: {e}") from e
        return (response.choices[0].message.content or "").strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in one call, returned in input order.

        Batched rather than one-at-a-time because extraction embeds every
        statement in a session -- one round-trip per statement was the
        dominant cost of `chatmem extract`.
        """
        if not self._config.embedding_model:
            raise LLMConfigError(
                "llm.embedding_model is not set in config.yaml -- set it to an embedding "
                "model id served by your LM Studio instance."
            )
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self._config.embedding_model, input=texts
            )
        except OpenAIError as e:
            raise LLMResponseError(f"embedding request failed: {e}") from e
        # The API doesn't promise response order matches input order.
        by_index = {d.index: list(d.embedding) for d in response.data}
        if len(by_index) != len(texts):
            raise LLMResponseError(
                f"expected {len(texts)} embeddings, got {len(by_index)}"
            )
        return [by_index[i] for i in range(len(texts))]
