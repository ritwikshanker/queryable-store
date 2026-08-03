"""SQLite storage: one file, no server.

Schema is created idempotently on connect. Re-ingesting a thread replaces
its messages and sessions outright (delete-then-insert) rather than trying
to reconcile row-by-row -- simple, and correct even if a code change alters
which rows get dropped between runs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from chatmem.identity import IdentityResolver
from chatmem.models import Message, Person, Session, Statement
from chatmem.parsers.text import normalize_alias

SCHEMA_VERSION = 3

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
    message_count INTEGER NOT NULL
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
    embedding          TEXT  -- JSON list of floats; NULL until embedded
);
CREATE INDEX statements_person ON statements(person_id);
CREATE INDEX statements_session ON statements(session_id);
"""

# Migrations applied in order to bring an existing DB from one
# schema_version to the next, so older databases keep working without a
# re-ingest. Each entry upgrades from its key to key+1.
_MIGRATIONS: dict[int, str] = {
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
        conn.executescript(_MIGRATIONS[version])
        version += 1
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(version),)
        )
    conn.commit()


# --- identity ----------------------------------------------------------


def save_identities(conn: sqlite3.Connection, resolver: IdentityResolver) -> None:
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
    conn.commit()


def relink_messages(conn: sqlite3.Connection, resolver: IdentityResolver) -> int:
    """Re-resolve every message's person_id from the current resolver state.

    Used by `chatmem relink` after config.yaml's identities change, without
    re-parsing the source archive.
    """
    rows = conn.execute("SELECT id, sender FROM messages").fetchall()
    updates = [(resolver.resolve(row["sender"]), row["id"]) for row in rows]
    # Any newly auto-created people must exist before messages can
    # reference them (person_id is a foreign key).
    save_identities(conn, resolver)
    conn.executemany("UPDATE messages SET person_id = ? WHERE id = ?", updates)
    conn.commit()
    return len(updates)


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


def list_sessions(
    conn: sqlite3.Connection, thread_id: str | None = None, limit: int | None = None
) -> list[Session]:
    sql = (
        "SELECT id, thread_id, start_seq, end_seq, start_ts, end_ts, message_count "
        "FROM sessions"
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
        )
        for r in rows
    ]


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
                json.dumps(s.embedding) if s.embedding is not None else None,
            )
            for s in statements
        ],
    )


def list_statements(
    conn: sqlite3.Connection,
    person_id: str | None = None,
    thread_id: str | None = None,
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
            embedding=json.loads(r["embedding"]) if r["embedding"] is not None else None,
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
