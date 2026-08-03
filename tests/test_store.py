from chatmem import store
from chatmem.config import IdentitySpec
from chatmem.identity import IdentityResolver
from chatmem.models import Message


def _ensure_people(conn, *person_ids):
    conn.executemany(
        "INSERT OR IGNORE INTO people (person_id, display_name, origin) VALUES (?, ?, 'auto')",
        [(pid, pid) for pid in person_ids],
    )


def _msg(id_, thread_id, sender, person_id, ts_ms, seq, text="hi", media_type=None):
    return Message(
        id=id_,
        thread_id=thread_id,
        sender=sender,
        person_id=person_id,
        timestamp_utc=f"1970-01-01T00:00:{ts_ms:02d}.000000Z",
        timestamp_ms=ts_ms,
        text=text,
        media_type=media_type,
        seq=seq,
    )


def test_connect_creates_schema_idempotently(tmp_path):
    db_path = tmp_path / "chatmem.db"
    conn1 = store.connect(db_path)
    conn1.close()
    # Reconnecting must not raise (schema already exists) and must not
    # wipe existing data.
    conn2 = store.connect(db_path)
    row = conn2.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    assert row["value"] == str(store.SCHEMA_VERSION)


def test_connect_migrates_v1_db_without_dropping_data(tmp_path):
    import sqlite3

    db_path = tmp_path / "chatmem.db"
    # Recreate a Phase-1 (schema_version=1) database by hand: same schema
    # minus the statements table that Phase 2 adds.
    v1_conn = sqlite3.connect(str(db_path))
    v1_conn.executescript(
        """
        CREATE TABLE people (person_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, origin TEXT NOT NULL);
        CREATE TABLE aliases (alias_norm TEXT PRIMARY KEY, raw_name TEXT NOT NULL, person_id TEXT NOT NULL);
        CREATE TABLE threads (thread_id TEXT PRIMARY KEY, title TEXT, participants TEXT, source TEXT NOT NULL, ingested_at TEXT NOT NULL);
        CREATE TABLE messages (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, sender TEXT NOT NULL, person_id TEXT NOT NULL, timestamp_utc TEXT NOT NULL, timestamp_ms INTEGER NOT NULL, text TEXT, media_type TEXT, seq INTEGER NOT NULL);
        CREATE TABLE sessions (id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL, start_seq INTEGER NOT NULL, end_seq INTEGER NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL, message_count INTEGER NOT NULL);
        CREATE TABLE session_participants (session_id INTEGER NOT NULL, person_id TEXT NOT NULL, PRIMARY KEY (session_id, person_id));
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta (key, value) VALUES ('schema_version', '1');
        INSERT INTO people (person_id, display_name, origin) VALUES ('pA', 'A', 'auto');
        """
    )
    v1_conn.commit()
    v1_conn.close()

    conn = store.connect(db_path)
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    assert row["value"] == str(store.SCHEMA_VERSION)
    # Existing data survived the migration.
    assert conn.execute("SELECT person_id FROM people").fetchone()["person_id"] == "pA"
    # The new table exists and is usable.
    conn.execute("SELECT COUNT(*) FROM statements")


def test_thread_and_message_round_trip(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", "Title", ["A", "B"], "instagram", "2024-01-01T00:00:00Z")
    _ensure_people(conn, "pA", "pB")
    msgs = [
        _msg("id0", "t1", "A", "pA", 0, 0, text="first"),
        _msg("id1", "t1", "B", "pB", 1, 1, text="second"),
    ]
    store.replace_thread_messages(conn, "t1", msgs)
    conn.commit()

    loaded = store.thread_messages(conn, "t1")
    assert [m.text for m in loaded] == ["first", "second"]
    assert [m.seq for m in loaded] == [0, 1]


def test_replace_thread_messages_replaces_not_duplicates(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")
    _ensure_people(conn, "pA")

    store.replace_thread_messages(conn, "t1", [_msg("id0", "t1", "A", "pA", 0, 0, text="v1")])
    conn.commit()
    store.replace_thread_messages(conn, "t1", [_msg("id1", "t1", "A", "pA", 0, 0, text="v2")])
    conn.commit()

    loaded = store.thread_messages(conn, "t1")
    assert len(loaded) == 1
    assert loaded[0].text == "v2"


def test_save_identities_and_relink(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")

    resolver = IdentityResolver([])
    pid_old = resolver.resolve("Old Name")
    store.save_identities(conn, resolver)
    store.replace_thread_messages(
        conn, "t1", [_msg("id0", "t1", "Old Name", pid_old, 0, 0, text="hi")]
    )
    conn.commit()

    # Now declare "Old Name" as an alias of a real identity and relink.
    resolver2 = IdentityResolver(
        [IdentitySpec(id="target", display_name="Real Name", aliases=["Old Name"])]
    )
    n = store.relink_messages(conn, resolver2)
    assert n == 1

    loaded = store.thread_messages(conn, "t1")
    assert loaded[0].person_id == "target"


def test_sessions_round_trip_with_participants(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")
    _ensure_people(conn, "pA", "pB")
    msgs = [
        _msg("id0", "t1", "A", "pA", 0, 0),
        _msg("id1", "t1", "B", "pB", 1, 1),
        _msg("id2", "t1", "A", "pA", 2, 2),
    ]
    store.replace_thread_messages(conn, "t1", msgs)
    by_seq = {m.seq: m for m in msgs}

    saved = store.replace_thread_sessions(conn, "t1", [(0, 2)], by_seq)
    conn.commit()
    assert len(saved) == 1
    assert saved[0].message_count == 3

    participants = conn.execute(
        "SELECT person_id FROM session_participants WHERE session_id = ?", (saved[0].id,)
    ).fetchall()
    assert {r["person_id"] for r in participants} == {"pA", "pB"}

    listed = store.list_sessions(conn, thread_id="t1")
    assert len(listed) == 1
    assert listed[0].start_seq == 0 and listed[0].end_seq == 2


def _statement(session_id, person_id="pA", thread_id="t1", text="hi", source_ids=None):
    from chatmem.models import Statement

    return Statement(
        id=None,
        person_id=person_id,
        session_id=session_id,
        thread_id=thread_id,
        text=text,
        source_message_ids=source_ids or [],
        start_ts="1970-01-01T00:00:00.000000Z",
        end_ts="1970-01-01T00:00:01.000000Z",
        created_at="1970-01-01T00:00:02.000000Z",
    )


def test_replace_session_statements_round_trip_and_replace(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")
    _ensure_people(conn, "pA")
    msgs = [_msg("id0", "t1", "A", "pA", 0, 0)]
    store.replace_thread_messages(conn, "t1", msgs)
    by_seq = {m.seq: m for m in msgs}
    [session] = store.replace_thread_sessions(conn, "t1", [(0, 0)], by_seq)
    conn.commit()

    store.replace_session_statements(
        conn, session.id, [_statement(session.id, source_ids=["id0"])]
    )
    conn.commit()
    loaded = store.list_statements(conn, person_id="pA")
    assert len(loaded) == 1
    assert loaded[0].source_message_ids == ["id0"]

    # Re-running replaces rather than duplicating.
    store.replace_session_statements(conn, session.id, [_statement(session.id, text="v2")])
    conn.commit()
    loaded = store.list_statements(conn, person_id="pA")
    assert len(loaded) == 1
    assert loaded[0].text == "v2"


def test_resolve_person_by_id_and_alias(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    resolver = IdentityResolver(
        [IdentitySpec(id="target", display_name="Real Name", aliases=["Old Name"])]
    )
    store.save_identities(conn, resolver)
    conn.commit()

    assert store.resolve_person(conn, "target") == "target"
    assert store.resolve_person(conn, "Old Name") == "target"
    assert store.resolve_person(conn, "old name") == "target"  # normalized match
    assert store.resolve_person(conn, "nobody") is None


def test_replace_thread_sessions_clears_old_ones(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")
    _ensure_people(conn, "pA")
    msgs = [_msg("id0", "t1", "A", "pA", 0, 0), _msg("id1", "t1", "A", "pA", 1, 1)]
    store.replace_thread_messages(conn, "t1", msgs)
    by_seq = {m.seq: m for m in msgs}

    store.replace_thread_sessions(conn, "t1", [(0, 0), (1, 1)], by_seq)
    conn.commit()
    store.replace_thread_sessions(conn, "t1", [(0, 1)], by_seq)
    conn.commit()

    listed = store.list_sessions(conn, thread_id="t1")
    assert len(listed) == 1
    assert listed[0].start_seq == 0 and listed[0].end_seq == 1
