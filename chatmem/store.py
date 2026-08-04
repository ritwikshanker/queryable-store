"""SQLite storage: one file, no server.

Schema is created idempotently on connect. Re-ingesting a thread replaces
its messages and sessions outright (delete-then-insert) rather than trying
to reconcile row-by-row -- simple, and correct even if a code change alters
which rows get dropped between runs.
"""

from __future__ import annotations

import json
import sqlite3
from array import array
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from chatmem.identity import IdentityResolver
from chatmem.models import Message, Person, Session, Statement
from chatmem.parsers.text import normalize_alias

SCHEMA_VERSION = 4

_SCHEMA_SQL = """
CREATE TABLE people (
    person_id    TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    origin       TEXT NOT NULL  -- 'config' | 'auto'
);

CREATE TABLE aliases (
    alias_norm TEXT PRIMARY KEY,
    raw_name   TEXT NOT NULL,
    person_id  TEXT NOT NULL REFERENCES people(person_id)
);

CREATE TABLE threads (
    thread_id    TEXT PRIMARY KEY,
    title        TEXT,
    participants TEXT,  -- JSON list of raw participant names
    source       TEXT NOT NULL,
    ingested_at  TEXT NOT NULL
);

CREATE TABLE messages (
    id            TEXT PRIMARY KEY,
    thread_id     TEXT NOT NULL REFERENCES threads(thread_id),
    sender        TEXT NOT NULL,
    person_id     TEXT NOT NULL REFERENCES people(person_id),
    timestamp_utc TEXT NOT NULL,
    timestamp_ms  INTEGER NOT NULL,
    text          TEXT,
    media_type    TEXT,
    seq           INTEGER NOT NULL
);
CREATE UNIQUE INDEX messages_thread_seq ON messages(thread_id, seq);
CREATE INDEX messages_thread_ts ON messages(thread_id, timestamp_ms);
CREATE INDEX messages_person ON messages(person_id);

CREATE TABLE sessions (
    id            INTEGER PRIMARY KEY,
    thread_id     TEXT NOT NULL REFERENCES threads(thread_id),
    start_seq     INTEGER NOT NULL,
    end_seq       INTEGER NOT NULL,
    start_ts      TEXT NOT NULL,
    end_ts        TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    extracted_at  TEXT  -- NULL until `chatmem extract` processes the session
);
CREATE INDEX sessions_thread ON sessions(thread_id, start_seq);

CREATE TABLE session_participants (
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    person_id  TEXT NOT NULL REFERENCES people(person_id),
    PRIMARY KEY (session_id, person_id)
);

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE statements (
    id                 INTEGER PRIMARY KEY,
    person_id          TEXT NOT NULL REFERENCES people(person_id),
    session_id         INTEGER NOT NULL REFERENCES sessions(id),
    thread_id          TEXT NOT NULL REFERENCES threads(thread_id),
    text               TEXT NOT NULL,
    source_message_ids TEXT NOT NULL,  -- JSON list of messages.id
    start_ts           TEXT NOT NULL,
    end_ts             TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    embedding          BLOB  -- packed float32 vector; NULL until embedded
);
CREATE INDEX statements_person ON statements(person_id);
CREATE INDEX statements_session ON statements(session_id);
"""


def _pack_embedding(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def _unpack_embedding(blob: bytes) -> list[float]:
    return list(array("f", blob))


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """v3 -> v4: statements.embedding JSON text -> packed float32 BLOB, and
    sessions.extracted_at backfilled from existing statements so already-
    extracted sessions are recognized by extract's resume logic.

    Rebuilds the statements table (SQLite can't alter a column's type), so
    foreign key enforcement is suspended for the rename/drop dance. The
    pragma only takes effect outside a transaction, hence the explicit
    commit/BEGIN ordering -- the rebuild itself is atomic.
    """
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN")
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN extracted_at TEXT")
        conn.execute("ALTER TABLE statements RENAME TO statements_old")
        conn.execute(
            """
            CREATE TABLE statements (
                id                 INTEGER PRIMARY KEY,
                person_id          TEXT NOT NULL REFERENCES people(person_id),
                session_id         INTEGER NOT NULL REFERENCES sessions(id),
                thread_id          TEXT NOT NULL REFERENCES threads(thread_id),
                text               TEXT NOT NULL,
                source_message_ids TEXT NOT NULL,
                start_ts           TEXT NOT NULL,
                end_ts             TEXT NOT NULL,
                created_at         TEXT NOT NULL,
                embedding          BLOB
            )
            """
        )
        for row in conn.execute(
            "SELECT id, person_id, session_id, thread_id, text, source_message_ids, "
            "start_ts, end_ts, created_at, embedding FROM statements_old"
        ).fetchall():
            embedding = row["embedding"]
            blob = _pack_embedding(json.loads(embedding)) if embedding is not None else None
            conn.execute(
                "INSERT INTO statements (id, person_id, session_id, thread_id, text, "
                "source_message_ids, start_ts, end_ts, created_at, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"],
                    row["person_id"],
                    row["session_id"],
                    row["thread_id"],
                    row["text"],
                    row["source_message_ids"],
                    row["start_ts"],
                    row["end_ts"],
                    row["created_at"],
                    blob,
                ),
            )
        conn.execute("DROP TABLE statements_old")
        conn.execute("CREATE INDEX statements_person ON statements(person_id)")
        conn.execute("CREATE INDEX statements_session ON statements(session_id)")
        conn.execute(
            "UPDATE sessions SET extracted_at = "
            "(SELECT MAX(created_at) FROM statements WHERE statements.session_id = sessions.id) "
            "WHERE id IN (SELECT DISTINCT session_id FROM statements)"
        )
    except Exception:
        conn.rollback()
        conn.execute("PRAGMA foreign_keys=ON")
        raise
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")

# Migrations applied in order to bring an existing DB from one
# schema_version to the next, so older databases keep working without a
# re-ingest. Each entry upgrades from its key to key+1. A string is run as
# a script; a callable is used when the step needs to transform data.
_MIGRATIONS: dict[int, str | Callable[[sqlite3.Connection], None]] = {
    1: """
    CREATE TABLE statements (
        id                 INTEGER PRIMARY KEY,
        person_id          TEXT NOT NULL REFERENCES people(person_id),
        session_id         INTEGER NOT NULL REFERENCES sessions(id),
        thread_id          TEXT NOT NULL REFERENCES threads(thread_id),
        text               TEXT NOT NULL,
        source_message_ids TEXT NOT NULL,
        start_ts           TEXT NOT NULL,
        end_ts             TEXT NOT NULL,
        created_at         TEXT NOT NULL
    );
    CREATE INDEX statements_person ON statements(person_id);
    CREATE INDEX statements_session ON statements(session_id);
    """,
    2: "ALTER TABLE statements ADD COLUMN embedding TEXT;",
    3: _migrate_v3_to_v4,
}


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
    ).fetchone()
    if exists is None:
        conn.executescript(_SCHEMA_SQL)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        return

    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    version = int(row["value"]) if row else 1
    while version < SCHEMA_VERSION:
        step = _MIGRATIONS[version]
        if callable(step):
            step(conn)
        else:
            conn.executescript(step)
        version += 1
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(version),)
        )
        conn.commit()


# --- meta ----------------------------------------------------------------


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


# --- identity ----------------------------------------------------------


def save_identities(
    conn: sqlite3.Connection, resolver: IdentityResolver, commit: bool = True
) -> None:
    people = resolver.people()
    conn.executemany(
        "INSERT INTO people (person_id, display_name, origin) VALUES (?, ?, ?) "
        "ON CONFLICT(person_id) DO UPDATE SET display_name=excluded.display_name, "
        "origin=excluded.origin",
        [(p.person_id, p.display_name, p.origin) for p in people],
    )
    conn.executemany(
        "INSERT INTO aliases (alias_norm, raw_name, person_id) VALUES (?, ?, ?) "
        "ON CONFLICT(alias_norm) DO UPDATE SET raw_name=excluded.raw_name, "
        "person_id=excluded.person_id",
        resolver.alias_rows(),
    )
    if commit:
        conn.commit()


@dataclass(frozen=True)
class RelinkResult:
    messages: int
    statements: int
    # Sessions whose statements could not be re-attributed because one old
    # person split into several new ones -- re-extract those to fix them.
    ambiguous_session_ids: list[int]


def relink_messages(conn: sqlite3.Connection, resolver: IdentityResolver) -> RelinkResult:
    """Re-resolve every message's person_id from the current resolver state.

    Used by `chatmem relink` after config.yaml's identities change, without
    re-parsing the source archive. Everything derived from person_id is
    rebuilt too: session_participants (which `extract` uses to decide whether
    the target took part) and statements.person_id (which `query` filters on),
    since leaving those stale would silently hide already-extracted facts.
    """
    rows = conn.execute("SELECT id, sender, person_id FROM messages").fetchall()
    updates: list[tuple[str, str]] = []
    remap: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        new_pid = resolver.resolve(row["sender"])
        updates.append((new_pid, row["id"]))
        if new_pid != row["person_id"]:
            remap[row["person_id"]].add(new_pid)

    # Any newly auto-created people must exist before messages can
    # reference them (person_id is a foreign key).
    save_identities(conn, resolver, commit=False)
    conn.executemany("UPDATE messages SET person_id = ? WHERE id = ?", updates)

    # session_participants is a pure function of the new message rows.
    conn.execute("DELETE FROM session_participants")
    conn.execute(
        "INSERT INTO session_participants (session_id, person_id) "
        "SELECT DISTINCT s.id, m.person_id FROM sessions s "
        "JOIN messages m ON m.thread_id = s.thread_id "
        "AND m.seq BETWEEN s.start_seq AND s.end_seq"
    )

    n_statements = 0
    ambiguous: list[int] = []
    for old_pid, new_pids in remap.items():
        if len(new_pids) == 1:
            cur = conn.execute(
                "UPDATE statements SET person_id = ? WHERE person_id = ?",
                (next(iter(new_pids)), old_pid),
            )
            n_statements += cur.rowcount
        else:
            # A split: which new person a statement belongs to can't be
            # decided without re-reading the transcript.
            ambiguous.extend(
                r["session_id"]
                for r in conn.execute(
                    "SELECT DISTINCT session_id FROM statements WHERE person_id = ? "
                    "ORDER BY session_id",
                    (old_pid,),
                ).fetchall()
            )

    conn.commit()
    return RelinkResult(
        messages=len(updates), statements=n_statements, ambiguous_session_ids=sorted(ambiguous)
    )


def all_people(conn: sqlite3.Connection) -> list[Person]:
    rows = conn.execute("SELECT person_id, display_name, origin FROM people").fetchall()
    return [Person(person_id=r["person_id"], display_name=r["display_name"], origin=r["origin"]) for r in rows]


def resolve_person(conn: sqlite3.Connection, value: str) -> str | None:
    """Resolve a person_id or alias against already-ingested data.

    Used by commands (like `extract`) that run after `ingest` and need
    config.yaml's `target` resolved without re-parsing the source archive or
    rebuilding an IdentityResolver.
    """
    row = conn.execute("SELECT person_id FROM people WHERE person_id = ?", (value,)).fetchone()
    if row is not None:
        return row["person_id"]
    row = conn.execute(
        "SELECT person_id FROM aliases WHERE alias_norm = ?", (normalize_alias(value),)
    ).fetchone()
    return row["person_id"] if row is not None else None


# --- threads / messages --------------------------------------------------


def upsert_thread(
    conn: sqlite3.Connection,
    thread_id: str,
    title: str | None,
    participants: list[str],
    source: str,
    ingested_at: str,
) -> None:
    conn.execute(
        "INSERT INTO threads (thread_id, title, participants, source, ingested_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(thread_id) DO UPDATE SET title=excluded.title, "
        "participants=excluded.participants, source=excluded.source, "
        "ingested_at=excluded.ingested_at",
        (thread_id, title, json.dumps(participants), source, ingested_at),
    )


def replace_thread_messages(conn: sqlite3.Connection, thread_id: str, messages: list[Message]) -> None:
    conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
    conn.executemany(
        "INSERT INTO messages "
        "(id, thread_id, sender, person_id, timestamp_utc, timestamp_ms, text, media_type, seq) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                m.id,
                m.thread_id,
                m.sender,
                m.person_id,
                m.timestamp_utc,
                m.timestamp_ms,
                m.text,
                m.media_type,
                m.seq,
            )
            for m in messages
        ],
    )


def thread_messages(conn: sqlite3.Connection, thread_id: str) -> list[Message]:
    rows = conn.execute(
        "SELECT id, thread_id, sender, person_id, timestamp_utc, timestamp_ms, text, "
        "media_type, seq FROM messages WHERE thread_id = ? ORDER BY seq ASC",
        (thread_id,),
    ).fetchall()
    return [
        Message(
            id=r["id"],
            thread_id=r["thread_id"],
            sender=r["sender"],
            person_id=r["person_id"],
            timestamp_utc=r["timestamp_utc"],
            timestamp_ms=r["timestamp_ms"],
            text=r["text"],
            media_type=r["media_type"],
            seq=r["seq"],
        )
        for r in rows
    ]


def all_thread_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT thread_id FROM threads ORDER BY thread_id").fetchall()
    return [r["thread_id"] for r in rows]


# --- sessions --------------------------------------------------------------


def replace_thread_sessions(
    conn: sqlite3.Connection,
    thread_id: str,
    ranges: list[tuple[int, int]],
    messages_by_seq: dict[int, Message],
) -> list[Session]:
    """Delete and rebuild sessions (and session_participants) for a thread.

    `ranges` are inclusive (start_seq, end_seq) pairs from sessionize().
    `messages_by_seq` must cover every seq referenced by `ranges`.
    """
    old_ids = [
        r["id"]
        for r in conn.execute("SELECT id FROM sessions WHERE thread_id = ?", (thread_id,)).fetchall()
    ]
    if old_ids:
        conn.executemany(
            "DELETE FROM session_participants WHERE session_id = ?", [(i,) for i in old_ids]
        )
    # statements.session_id is a NOT NULL foreign key, so any statements
    # still pointing at these sessions must go first or the delete fails.
    # Callers that want to keep them (re-ingest) go through
    # replace_thread_content, which snapshots and restores them.
    conn.execute("DELETE FROM statements WHERE thread_id = ?", (thread_id,))
    conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))

    saved: list[Session] = []
    for start_seq, end_seq in ranges:
        start_ts = messages_by_seq[start_seq].timestamp_utc
        end_ts = messages_by_seq[end_seq].timestamp_utc
        count = end_seq - start_seq + 1
        cur = conn.execute(
            "INSERT INTO sessions (thread_id, start_seq, end_seq, start_ts, end_ts, message_count) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, start_seq, end_seq, start_ts, end_ts, count),
        )
        session_id = cur.lastrowid
        person_ids = {
            messages_by_seq[seq].person_id
            for seq in range(start_seq, end_seq + 1)
            if seq in messages_by_seq
        }
        conn.executemany(
            "INSERT INTO session_participants (session_id, person_id) VALUES (?, ?)",
            [(session_id, pid) for pid in person_ids],
        )
        saved.append(
            Session(
                id=session_id,
                thread_id=thread_id,
                start_seq=start_seq,
                end_seq=end_seq,
                start_ts=start_ts,
                end_ts=end_ts,
                message_count=count,
            )
        )
    return saved


def replace_thread_content(
    conn: sqlite3.Connection,
    thread_id: str,
    messages: list[Message],
    ranges: list[tuple[int, int]],
    messages_by_seq: dict[int, Message],
) -> tuple[list[Session], int]:
    """Replace a thread's messages and sessions, preserving statements whose
    session came through the rebuild unchanged.

    Extraction is the expensive part of the pipeline, so re-ingesting an
    archive that hasn't changed must not throw its output away. A session is
    considered the same session if its range, bounds and size all match and
    every message its statements cite still exists -- message ids are content
    hashes, so any edit to the cited text breaks the match and the session is
    left unextracted for `chatmem extract` to redo.

    Returns the saved sessions and how many statements were carried over.
    """
    old_sessions = {
        (r["start_seq"], r["end_seq"], r["start_ts"], r["end_ts"], r["message_count"]): (
            r["id"],
            r["extracted_at"],
        )
        for r in conn.execute(
            "SELECT id, start_seq, end_seq, start_ts, end_ts, message_count, extracted_at "
            "FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchall()
    }
    old_statements: dict[int, list[Statement]] = defaultdict(list)
    for s in list_statements(conn, thread_id=thread_id):
        old_statements[s.session_id].append(s)

    replace_thread_messages(conn, thread_id, messages)
    saved = replace_thread_sessions(conn, thread_id, ranges, messages_by_seq)

    live_ids = {m.id for m in messages}
    preserved = 0
    for session in saved:
        key = (
            session.start_seq,
            session.end_seq,
            session.start_ts,
            session.end_ts,
            session.message_count,
        )
        match = old_sessions.get(key)
        if match is None:
            continue
        old_id, extracted_at = match
        carried = old_statements.get(old_id, [])
        if any(mid not in live_ids for s in carried for mid in s.source_message_ids):
            continue
        if carried:
            replace_session_statements(
                conn,
                session.id,
                [replace(s, id=None, session_id=session.id) for s in carried],
            )
            preserved += len(carried)
        if extracted_at is not None:
            mark_session_extracted(conn, session.id, extracted_at)

    return saved, preserved


def list_sessions(
    conn: sqlite3.Connection, thread_id: str | None = None, limit: int | None = None
) -> list[Session]:
    sql = (
        "SELECT id, thread_id, start_seq, end_seq, start_ts, end_ts, message_count, "
        "extracted_at FROM sessions"
    )
    params: list[object] = []
    if thread_id is not None:
        sql += " WHERE thread_id = ?"
        params.append(thread_id)
    sql += " ORDER BY thread_id, start_seq"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        Session(
            id=r["id"],
            thread_id=r["thread_id"],
            start_seq=r["start_seq"],
            end_seq=r["end_seq"],
            start_ts=r["start_ts"],
            end_ts=r["end_ts"],
            message_count=r["message_count"],
            extracted_at=r["extracted_at"],
        )
        for r in rows
    ]


def mark_session_extracted(conn: sqlite3.Connection, session_id: int, when: str) -> None:
    conn.execute("UPDATE sessions SET extracted_at = ? WHERE id = ?", (when, session_id))


def participants_by_session(conn: sqlite3.Connection) -> dict[int, set[str]]:
    """Every session's participants in one query -- `extract` needs this for
    each session it considers, and one query per session added up."""
    result: dict[int, set[str]] = defaultdict(set)
    for r in conn.execute("SELECT session_id, person_id FROM session_participants").fetchall():
        result[r["session_id"]].add(r["person_id"])
    return result


def session_participant_ids(conn: sqlite3.Connection, session_id: int) -> set[str]:
    rows = conn.execute(
        "SELECT person_id FROM session_participants WHERE session_id = ?", (session_id,)
    ).fetchall()
    return {r["person_id"] for r in rows}


def session_messages(conn: sqlite3.Connection, thread_id: str, start_seq: int, end_seq: int) -> list[Message]:
    rows = conn.execute(
        "SELECT id, thread_id, sender, person_id, timestamp_utc, timestamp_ms, text, "
        "media_type, seq FROM messages WHERE thread_id = ? AND seq BETWEEN ? AND ? "
        "ORDER BY seq ASC",
        (thread_id, start_seq, end_seq),
    ).fetchall()
    return [
        Message(
            id=r["id"],
            thread_id=r["thread_id"],
            sender=r["sender"],
            person_id=r["person_id"],
            timestamp_utc=r["timestamp_utc"],
            timestamp_ms=r["timestamp_ms"],
            text=r["text"],
            media_type=r["media_type"],
            seq=r["seq"],
        )
        for r in rows
    ]


# --- statements ----------------------------------------------------------


def replace_session_statements(
    conn: sqlite3.Connection, session_id: int, statements: list[Statement]
) -> None:
    """Delete and rebuild a session's statements, matching the delete-then-insert
    pattern used for messages and sessions, so re-running extract is idempotent."""
    conn.execute("DELETE FROM statements WHERE session_id = ?", (session_id,))
    conn.executemany(
        "INSERT INTO statements "
        "(person_id, session_id, thread_id, text, source_message_ids, start_ts, end_ts, "
        "created_at, embedding) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                s.person_id,
                s.session_id,
                s.thread_id,
                s.text,
                json.dumps(s.source_message_ids),
                s.start_ts,
                s.end_ts,
                s.created_at,
                _pack_embedding(s.embedding) if s.embedding is not None else None,
            )
            for s in statements
        ],
    )


def list_statements(
    conn: sqlite3.Connection,
    person_id: str | None = None,
    thread_id: str | None = None,
    session_id: int | None = None,
    limit: int | None = None,
) -> list[Statement]:
    sql = (
        "SELECT id, person_id, session_id, thread_id, text, source_message_ids, "
        "start_ts, end_ts, created_at, embedding FROM statements"
    )
    clauses = []
    params: list[object] = []
    if person_id is not None:
        clauses.append("person_id = ?")
        params.append(person_id)
    if thread_id is not None:
        clauses.append("thread_id = ?")
        params.append(thread_id)
    if session_id is not None:
        clauses.append("session_id = ?")
        params.append(session_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY start_ts ASC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [
        Statement(
            id=r["id"],
            person_id=r["person_id"],
            session_id=r["session_id"],
            thread_id=r["thread_id"],
            text=r["text"],
            source_message_ids=json.loads(r["source_message_ids"]),
            start_ts=r["start_ts"],
            end_ts=r["end_ts"],
            created_at=r["created_at"],
            embedding=_unpack_embedding(r["embedding"]) if r["embedding"] is not None else None,
        )
        for r in rows
    ]


def messages_by_ids(conn: sqlite3.Connection, message_ids: list[str]) -> list[Message]:
    """Fetch messages by id, in the given order -- used to quote a statement's
    source messages back for `chatmem query`."""
    if not message_ids:
        return []
    placeholders = ",".join("?" for _ in message_ids)
    rows = conn.execute(
        "SELECT id, thread_id, sender, person_id, timestamp_utc, timestamp_ms, text, "
        f"media_type, seq FROM messages WHERE id IN ({placeholders})",
        message_ids,
    ).fetchall()
    by_id = {
        r["id"]: Message(
            id=r["id"],
            thread_id=r["thread_id"],
            sender=r["sender"],
            person_id=r["person_id"],
            timestamp_utc=r["timestamp_utc"],
            timestamp_ms=r["timestamp_ms"],
            text=r["text"],
            media_type=r["media_type"],
            seq=r["seq"],
        )
        for r in rows
    }
    return [by_id[mid] for mid in message_ids if mid in by_id]


# --- stats -------------------------------------------------------------


def thread_stats(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT t.thread_id, t.title, COUNT(m.id) AS message_count, "
        "MIN(m.timestamp_utc) AS first_ts, MAX(m.timestamp_utc) AS last_ts "
        "FROM threads t LEFT JOIN messages m ON m.thread_id = t.thread_id "
        "GROUP BY t.thread_id ORDER BY t.thread_id"
    ).fetchall()


def person_stats(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT p.person_id, p.display_name, p.origin, COUNT(m.id) AS message_count "
        "FROM people p LEFT JOIN messages m ON m.person_id = p.person_id "
        "GROUP BY p.person_id ORDER BY message_count DESC"
    ).fetchall()
