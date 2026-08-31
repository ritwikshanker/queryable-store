"""Configuration loading.

config.yaml is the single source of truth for everything user-specific
(db path, target participant, identity aliases, model ids). A missing file
is not an error — every field has a default — but a malformed one is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


class ConfigError(ValueError):
    """Raised for a malformed config.yaml (bad shape, duplicate alias, ...)."""


@dataclass(frozen=True)
class IdentitySpec:
    id: str
    display_name: str
    aliases: list[str]


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    chat_model: str = ""
    embedding_model: str = ""
    timeout_seconds: float = 60.0
    max_retries: int = 3
    # Passed straight through to the chat endpoint when set. Extraction and
    # classification are mechanical tasks, so a reasoning model burning tokens
    # on them is pure latency -- "none" measured ~3x faster on qwen3 via
    # Ollama. Left None by default because not every server accepts the
    # parameter at all.
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class SessionizeConfig:
    idle_gap_minutes: float = 120.0
    max_messages: int = 40


@dataclass(frozen=True)
class ExtractionConfig:
    validation_pass: bool = False
    # Cosine similarity at or above which a newly extracted statement is
    # treated as a restatement of one already stored and dropped. 0 (or less)
    # disables the check.
    dedup_threshold: float = 0.92


@dataclass(frozen=True)
class Config:
    db_path: Path = Path("data/chatmem.db")
    target: str | None = None
    identities: list[IdentitySpec] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sessionize: SessionizeConfig = field(default_factory=SessionizeConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    def with_overrides(
        self, *, target: str | None = None, db_path: Path | None = None
    ) -> "Config":
        """Return a copy with CLI-level overrides applied (e.g. --target, --db)."""
        return Config(
            db_path=db_path if db_path is not None else self.db_path,
            target=target if target is not None else self.target,
            identities=self.identities,
            llm=self.llm,
            sessionize=self.sessionize,
            extraction=self.extraction,
        )


def _parse_identities(raw: Any) -> list[IdentitySpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError("'identities' must be a list")

    specs: list[IdentitySpec] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"identities[{i}] must be a mapping")
        pid = entry.get("id")
        if not pid or not isinstance(pid, str):
            raise ConfigError(f"identities[{i}] is missing a string 'id'")
        if pid in seen_ids:
            raise ConfigError(f"duplicate identity id {pid!r} in 'identities'")
        seen_ids.add(pid)

        display_name = entry.get("display_name", pid)
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ConfigError(f"identities[{i}].aliases must be a list of strings")

        specs.append(IdentitySpec(id=pid, display_name=display_name, aliases=list(aliases)))

    # Duplicate alias across two different ids is a config error, named explicitly.
    # (Identity normalization happens in chatmem.identity; here we only guard the
    # cheap case of the exact same raw string appearing under two ids, which
    # covers the common copy-paste mistake without duplicating normalization
    # logic in this module.)
    owner_by_alias: dict[str, str] = {}
    for spec in specs:
        for alias in spec.aliases:
            if alias in owner_by_alias and owner_by_alias[alias] != spec.id:
                raise ConfigError(
                    f"alias {alias!r} is declared under both "
                    f"{owner_by_alias[alias]!r} and {spec.id!r}"
                )
            owner_by_alias[alias] = spec.id

    return specs


def _parse_llm(raw: Any) -> LLMConfig:
    if raw is None:
        return LLMConfig()
    if not isinstance(raw, dict):
        raise ConfigError("'llm' must be a mapping")
    defaults = LLMConfig()
    return LLMConfig(
        base_url=raw.get("base_url", defaults.base_url),
        api_key=raw.get("api_key", defaults.api_key),
        chat_model=raw.get("chat_model", defaults.chat_model),
        embedding_model=raw.get("embedding_model", defaults.embedding_model),
        timeout_seconds=raw.get("timeout_seconds", defaults.timeout_seconds),
        max_retries=raw.get("max_retries", defaults.max_retries),
        reasoning_effort=raw.get("reasoning_effort", defaults.reasoning_effort),
    )


def _parse_sessionize(raw: Any) -> SessionizeConfig:
    if raw is None:
        return SessionizeConfig()
    if not isinstance(raw, dict):
        raise ConfigError("'sessionize' must be a mapping")
    defaults = SessionizeConfig()
    return SessionizeConfig(
        idle_gap_minutes=raw.get("idle_gap_minutes", defaults.idle_gap_minutes),
        max_messages=raw.get("max_messages", defaults.max_messages),
    )


def _parse_extraction(raw: Any) -> ExtractionConfig:
    if raw is None:
        return ExtractionConfig()
    if not isinstance(raw, dict):
        raise ConfigError("'extraction' must be a mapping")
    threshold = raw.get("dedup_threshold", 0.92)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ConfigError("'extraction.dedup_threshold' must be a number")
    if threshold > 1:
        raise ConfigError("'extraction.dedup_threshold' must be <= 1 (cosine similarity)")
    return ExtractionConfig(
        validation_pass=bool(raw.get("validation_pass", False)),
        dedup_threshold=float(threshold),
    )


def load_config(path: Path | None = None) -> Config:
    """Load config.yaml, or return all-defaults if the file does not exist."""
    path = path if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return Config()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: invalid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")

    db_path = Path(raw.get("db_path", "data/chatmem.db"))
    target = raw.get("target")
    if target is not None and not isinstance(target, str):
        raise ConfigError("'target' must be a string or null")

    return Config(
        db_path=db_path,
        target=target,
        identities=_parse_identities(raw.get("identities")),
        llm=_parse_llm(raw.get("llm")),
        sessionize=_parse_sessionize(raw.get("sessionize")),
        extraction=_parse_extraction(raw.get("extraction")),
    )
