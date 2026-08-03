"""Identity resolution: map raw sender display names to a stable person_id.

One person may hold several accounts with different display names, possibly
in different threads. config.yaml's `identities:` block is the single source
of truth for that mapping (see config.example.yaml). Anything not declared
there is auto-created as its own person the first time it is seen, so
resolve() never fails -- ingest always completes, and IdentityResolver just
collects a warning for anything it had to invent so the user can fold it
into config.yaml (or a real merge) afterwards.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from chatmem.config import ConfigError, IdentitySpec
from chatmem.models import Person
from chatmem.parsers.text import normalize_alias

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(norm: str) -> str:
    slug = _SLUG_RE.sub("-", norm).strip("-")
    return slug or "person"


@dataclass
class NewSender:
    """An observed sender not covered by any declared identity."""

    raw_name: str
    alias_norm: str
    person_id: str


class IdentityResolver:
    """Resolves raw sender names to person_id, auto-creating on first sight."""

    def __init__(self, identities: list[IdentitySpec]):
        self._people: dict[str, Person] = {}
        self._alias_to_pid: dict[str, str] = {}
        self._alias_raw: dict[str, str] = {}  # alias_norm -> a representative raw string
        self._new_senders: dict[str, NewSender] = {}  # keyed by alias_norm

        for spec in identities:
            self._people[spec.id] = Person(
                person_id=spec.id, display_name=spec.display_name, origin="config"
            )
            # The display name itself is implicitly a valid alias, even if
            # the author forgot to list it explicitly.
            aliases = set(spec.aliases) | {spec.display_name}
            for alias in aliases:
                norm = normalize_alias(alias)
                existing = self._alias_to_pid.get(norm)
                if existing is not None and existing != spec.id:
                    raise ConfigError(
                        f"alias {alias!r} (normalized {norm!r}) is claimed by both "
                        f"identity {existing!r} and {spec.id!r}"
                    )
                self._alias_to_pid[norm] = spec.id
                self._alias_raw.setdefault(norm, alias)

    def resolve(self, raw_sender_name: str) -> str:
        """Return the person_id for a raw sender name, auto-creating if new."""
        norm = normalize_alias(raw_sender_name)

        pid = self._alias_to_pid.get(norm)
        if pid is not None:
            return pid

        existing_new = self._new_senders.get(norm)
        if existing_new is not None:
            return existing_new.person_id

        pid = self._fresh_slug(norm)
        self._people[pid] = Person(person_id=pid, display_name=raw_sender_name, origin="auto")
        self._alias_to_pid[norm] = pid
        self._alias_raw.setdefault(norm, raw_sender_name)
        self._new_senders[norm] = NewSender(raw_name=raw_sender_name, alias_norm=norm, person_id=pid)
        return pid

    def _fresh_slug(self, norm: str) -> str:
        base = _slugify(norm)
        if base not in self._people:
            return base
        # Collision between two different normalized names that happen to
        # slugify the same way (e.g. differing only in punctuation): fall
        # back to a short content hash to keep ids stable across runs.
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:6]
        return f"{base}-{digest}"

    def resolve_target(self, value: str) -> str | None:
        """Resolve a --target/config `target` value to a person_id.

        Accepts a person id or any alias (declared, or seen so far via
        resolve() during this run). Returns None rather than raising, since
        the target person may simply not appear in the archive being
        ingested right now -- callers should treat a miss as a warning, not
        a hard failure.
        """
        if value in self._people:
            return value
        return self._alias_to_pid.get(normalize_alias(value))

    def people(self) -> list[Person]:
        return list(self._people.values())

    def alias_rows(self) -> list[tuple[str, str, str]]:
        """All known (alias_norm, representative_raw_name, person_id) rows."""
        return [
            (norm, self._alias_raw.get(norm, norm), pid)
            for norm, pid in self._alias_to_pid.items()
        ]

    def new_senders(self) -> list[NewSender]:
        """Senders auto-created during this run (not in declared identities)."""
        return list(self._new_senders.values())

    def new_senders_yaml(self) -> str:
        """A paste-ready `identities:` block for every auto-created sender."""
        lines = []
        for ns in self.new_senders():
            lines.append(f"  - id: {ns.person_id}")
            lines.append(f"    display_name: {ns.raw_name!r}")
            lines.append(f"    aliases: [{ns.raw_name!r}]")
        return "\n".join(lines)
