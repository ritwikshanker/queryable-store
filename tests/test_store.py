import pytest

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


def test_connect_migrates_v2_db_adding_embedding_column(tmp_path):
    import sqlite3

    db_path = tmp_path / "chatmem.db"
    # A Phase-2 (schema_version=2) database: statements table exists, but
    # without the embedding column Phase 3 adds.
    v2_conn = sqlite3.connect(str(db_path))
    v2_conn.executescript(
        """
        CREATE TABLE people (person_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, origin TEXT NOT NULL);
        CREATE TABLE aliases (alias_norm TEXT PRIMARY KEY, raw_name TEXT NOT NULL, person_id TEXT NOT NULL);
        CREATE TABLE threads (thread_id TEXT PRIMARY KEY, title TEXT, participants TEXT, source TEXT NOT NULL, ingested_at TEXT NOT NULL);
        CREATE TABLE messages (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, sender TEXT NOT NULL, person_id TEXT NOT NULL, timestamp_utc TEXT NOT NULL, timestamp_ms INTEGER NOT NULL, text TEXT, media_type TEXT, seq INTEGER NOT NULL);
        CREATE TABLE sessions (id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL, start_seq INTEGER NOT NULL, end_seq INTEGER NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL, message_count INTEGER NOT NULL);
        CREATE TABLE session_participants (session_id INTEGER NOT NULL, person_id TEXT NOT NULL, PRIMARY KEY (session_id, person_id));
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE statements (id INTEGER PRIMARY KEY, person_id TEXT NOT NULL, session_id INTEGER NOT NULL, thread_id TEXT NOT NULL, text TEXT NOT NULL, source_message_ids TEXT NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL, created_at TEXT NOT NULL);
        INSERT INTO meta (key, value) VALUES ('schema_version', '2');
        INSERT INTO people (person_id, display_name, origin) VALUES ('pA', 'A', 'auto');
        INSERT INTO statements (person_id, session_id, thread_id, text, source_message_ids, start_ts, end_ts, created_at)
            VALUES ('pA', 1, 't1', 'hi', '[]', '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z');
        """
    )
    v2_conn.commit()
    v2_conn.close()

    conn = store.connect(db_path)
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    assert row["value"] == str(store.SCHEMA_VERSION)
    # Existing statement survived the migration, with embedding NULL.
    stmt_row = conn.execute("SELECT text, embedding FROM statements").fetchone()
    assert stmt_row["text"] == "hi"
    assert stmt_row["embedding"] is None


def test_connect_migrates_v3_db_packing_embeddings_and_backfilling_extracted_at(tmp_path):
    import json
    import sqlite3

    db_path = tmp_path / "chatmem.db"
    vec = [0.5, -0.25, 0.125]
    # A Phase-3 (schema_version=3) database: embeddings are JSON text and
    # sessions have no extracted_at column.
    v3_conn = sqlite3.connect(str(db_path))
    v3_conn.executescript(
        """
        CREATE TABLE people (person_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, origin TEXT NOT NULL);
        CREATE TABLE aliases (alias_norm TEXT PRIMARY KEY, raw_name TEXT NOT NULL, person_id TEXT NOT NULL);
        CREATE TABLE threads (thread_id TEXT PRIMARY KEY, title TEXT, participants TEXT, source TEXT NOT NULL, ingested_at TEXT NOT NULL);
        CREATE TABLE messages (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, sender TEXT NOT NULL, person_id TEXT NOT NULL, timestamp_utc TEXT NOT NULL, timestamp_ms INTEGER NOT NULL, text TEXT, media_type TEXT, seq INTEGER NOT NULL);
        CREATE TABLE sessions (id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL, start_seq INTEGER NOT NULL, end_seq INTEGER NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL, message_count INTEGER NOT NULL);
        CREATE TABLE session_participants (session_id INTEGER NOT NULL, person_id TEXT NOT NULL, PRIMARY KEY (session_id, person_id));
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE statements (id INTEGER PRIMARY KEY, person_id TEXT NOT NULL, session_id INTEGER NOT NULL, thread_id TEXT NOT NULL, text TEXT NOT NULL, source_message_ids TEXT NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL, created_at TEXT NOT NULL, embedding TEXT);
        INSERT INTO meta (key, value) VALUES ('schema_version', '3');
        INSERT INTO people (person_id, display_name, origin) VALUES ('pA', 'A', 'auto');
        INSERT INTO threads (thread_id, title, participants, source, ingested_at) VALUES ('t1', 'T', '[]', 'instagram', '1970-01-01T00:00:00.000000Z');
        INSERT INTO sessions (id, thread_id, start_seq, end_seq, start_ts, end_ts, message_count)
            VALUES (1, 't1', 0, 1, '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:01.000000Z', 2);
        INSERT INTO sessions (id, thread_id, start_seq, end_seq, start_ts, end_ts, message_count)
            VALUES (2, 't1', 2, 3, '1970-01-01T00:00:02.000000Z', '1970-01-01T00:00:03.000000Z', 2);
        """
    )
    v3_conn.execute(
        "INSERT INTO statements (person_id, session_id, thread_id, text, source_message_ids, "
        "start_ts, end_ts, created_at, embedding) VALUES ('pA', 1, 't1', 'hi', '[]', "
        "'1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z', "
        "'2024-05-05T00:00:00.000000Z', ?)",
        (json.dumps(vec),),
    )
    v3_conn.commit()
    v3_conn.close()

    conn = store.connect(db_path)
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    assert row["value"] == str(store.SCHEMA_VERSION)

    # The JSON embedding became a packed BLOB with the same values.
    stmt = store.list_statements(conn)[0]
    assert stmt.text == "hi"
    assert stmt.embedding == vec
    assert isinstance(
        conn.execute("SELECT embedding FROM statements").fetchone()["embedding"], bytes
    )

    # Sessions that already had statements are marked extracted; others aren't.
    by_id = {s.id: s for s in store.list_sessions(conn)}
    assert by_id[1].extracted_at == "2024-05-05T00:00:00.000000Z"
    assert by_id[2].extracted_at is None


def test_connect_migrates_v4_db_adding_topic_column_left_null(tmp_path):
    import sqlite3

    db_path = tmp_path / "chatmem.db"
    # A Phase-4 (schema_version=4) database: statements exist and are embedded,
    # but have no topic column.
    v4_conn = sqlite3.connect(str(db_path))
    v4_conn.executescript(
        """
        CREATE TABLE people (person_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, origin TEXT NOT NULL);
        CREATE TABLE aliases (alias_norm TEXT PRIMARY KEY, raw_name TEXT NOT NULL, person_id TEXT NOT NULL);
        CREATE TABLE threads (thread_id TEXT PRIMARY KEY, title TEXT, participants TEXT, source TEXT NOT NULL, ingested_at TEXT NOT NULL);
        CREATE TABLE messages (id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, sender TEXT NOT NULL, person_id TEXT NOT NULL, timestamp_utc TEXT NOT NULL, timestamp_ms INTEGER NOT NULL, text TEXT, media_type TEXT, seq INTEGER NOT NULL);
        CREATE TABLE sessions (id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL, start_seq INTEGER NOT NULL, end_seq INTEGER NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL, message_count INTEGER NOT NULL, extracted_at TEXT);
        CREATE TABLE session_participants (session_id INTEGER NOT NULL, person_id TEXT NOT NULL, PRIMARY KEY (session_id, person_id));
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE statements (id INTEGER PRIMARY KEY, person_id TEXT NOT NULL, session_id INTEGER NOT NULL, thread_id TEXT NOT NULL, text TEXT NOT NULL, source_message_ids TEXT NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL, created_at TEXT NOT NULL, embedding BLOB);
        INSERT INTO meta (key, value) VALUES ('schema_version', '4');
        INSERT INTO people (person_id, display_name, origin) VALUES ('pA', 'A', 'auto');
        INSERT INTO threads (thread_id, title, participants, source, ingested_at) VALUES ('t1', 'T', '[]', 'instagram', '1970-01-01T00:00:00.000000Z');
        INSERT INTO sessions (id, thread_id, start_seq, end_seq, start_ts, end_ts, message_count, extracted_at)
            VALUES (1, 't1', 0, 1, '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:01.000000Z', 2, '2024-05-05T00:00:00.000000Z');
        INSERT INTO statements (person_id, session_id, thread_id, text, source_message_ids, start_ts, end_ts, created_at, embedding)
            VALUES ('pA', 1, 't1', 'hi', '[]', '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z', '2024-05-05T00:00:00.000000Z', NULL);
        """
    )
    v4_conn.commit()
    v4_conn.close()

    conn = store.connect(db_path)
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    assert row["value"] == str(store.SCHEMA_VERSION)

    # The statement survived, and is left untagged so the first `digest` run
    # picks it up without needing --reclassify.
    stmt = store.list_statements(conn)[0]
    assert stmt.text == "hi"
    assert stmt.topic is None
    assert store.list_statements(conn, untagged=True) == [stmt]
    # Extraction state is untouched, so `extract` still resumes correctly.
    assert store.list_sessions(conn)[0].extracted_at == "2024-05-05T00:00:00.000000Z"


def test_set_statement_topics_assigns_by_id_and_untagged_filter_shrinks(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    _ensure_people(conn, "pA")
    conn.execute(
        "INSERT INTO threads (thread_id, title, participants, source, ingested_at) "
        "VALUES ('t1', 'T', '[]', 'instagram', '1970-01-01T00:00:00.000000Z')"
    )
    conn.execute(
        "INSERT INTO sessions (id, thread_id, start_seq, end_seq, start_ts, end_ts, message_count) "
        "VALUES (1, 't1', 0, 1, '1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:01.000000Z', 2)"
    )
    conn.executemany(
        "INSERT INTO statements (id, person_id, session_id, thread_id, text, source_message_ids, "
        "start_ts, end_ts, created_at) VALUES (?, 'pA', 1, 't1', ?, '[]', "
        "'1970-01-01T00:00:00.000000Z', '1970-01-01T00:00:00.000000Z', '2024-05-05T00:00:00.000000Z')",
        [(1, "a"), (2, "b")],
    )
    conn.commit()

    store.set_statement_topics(conn, {1: "family"})
    conn.commit()
    assert {s.id: s.topic for s in store.list_statements(conn)} == {1: "family", 2: None}
    assert [s.id for s in store.list_statements(conn, untagged=True)] == [2]

    # --reclassify resets everything so a changed taxonomy is applied in full.
    assert store.clear_statement_topics(conn) == 1
    conn.commit()
    assert len(store.list_statements(conn, untagged=True)) == 2


def test_pack_unpack_embedding_round_trip():
    vec = [0.5, -0.25, 0.125, 0.0]
    assert store._unpack_embedding(store._pack_embedding(vec)) == vec
    assert store._pack_embedding([]) == b""
    assert store._unpack_embedding(b"") == []


def test_meta_get_and_set(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    assert store.get_meta(conn, "embedding_model") is None
    store.set_meta(conn, "embedding_model", "model-a")
    assert store.get_meta(conn, "embedding_model") == "model-a"
    store.set_meta(conn, "embedding_model", "model-b")
    assert store.get_meta(conn, "embedding_model") == "model-b"


def test_mark_session_extracted(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", "T", [], "instagram", "1970-01-01T00:00:00.000000Z")
    _ensure_people(conn, "pA")
    msgs = [_msg("m0", "t1", "A", "pA", 0, 0), _msg("m1", "t1", "A", "pA", 1, 1)]
    store.replace_thread_messages(conn, "t1", msgs)
    sessions = store.replace_thread_sessions(conn, "t1", [(0, 1)], {m.seq: m for m in msgs})
    assert sessions[0].extracted_at is None

    store.mark_session_extracted(conn, sessions[0].id, "2024-01-01T00:00:00.000000Z")
    assert store.list_sessions(conn)[0].extracted_at == "2024-01-01T00:00:00.000000Z"


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
    result = store.relink_messages(conn, resolver2)
    assert result.messages == 1

    loaded = store.thread_messages(conn, "t1")
    assert loaded[0].person_id == "target"


def test_relink_rebuilds_participants_and_reattributes_statements(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")

    resolver = IdentityResolver([])
    pid_old = resolver.resolve("Old Name")
    store.save_identities(conn, resolver)
    msgs = [_msg("id0", "t1", "Old Name", pid_old, 0, 0)]
    store.replace_thread_messages(conn, "t1", msgs)
    [session] = store.replace_thread_sessions(conn, "t1", [(0, 0)], {m.seq: m for m in msgs})
    store.replace_session_statements(
        conn, session.id, [_statement(session.id, person_id=pid_old, source_ids=["id0"])]
    )
    conn.commit()

    resolver2 = IdentityResolver(
        [IdentitySpec(id="target", display_name="Real Name", aliases=["Old Name"])]
    )
    result = store.relink_messages(conn, resolver2)

    assert result.messages == 1
    assert result.statements == 1
    assert result.ambiguous_session_ids == []
    # The statement follows the message to the new person, so `query
    # --target target` still finds it.
    assert store.list_statements(conn, person_id="target")[0].source_message_ids == ["id0"]
    assert store.session_participant_ids(conn, session.id) == {"target"}


def test_relink_flags_sessions_when_one_person_splits(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")

    # Two different raw senders initially collapsed onto one person...
    resolver = IdentityResolver(
        [IdentitySpec(id="both", display_name="Both", aliases=["Ana", "Bea"])]
    )
    pid = resolver.resolve("Ana")
    store.save_identities(conn, resolver)
    msgs = [_msg("id0", "t1", "Ana", pid, 0, 0), _msg("id1", "t1", "Bea", pid, 1, 1)]
    store.replace_thread_messages(conn, "t1", msgs)
    [session] = store.replace_thread_sessions(conn, "t1", [(0, 1)], {m.seq: m for m in msgs})
    store.replace_session_statements(
        conn, session.id, [_statement(session.id, person_id=pid, source_ids=["id0"])]
    )
    conn.commit()

    # ...are now declared as two people. Which one the statement belongs to
    # can't be decided here, so the session is reported for re-extraction.
    resolver2 = IdentityResolver(
        [
            IdentitySpec(id="ana", display_name="Ana", aliases=["Ana"]),
            IdentitySpec(id="bea", display_name="Bea", aliases=["Bea"]),
        ]
    )
    result = store.relink_messages(conn, resolver2)

    assert result.statements == 0
    assert result.ambiguous_session_ids == [session.id]
    assert store.session_participant_ids(conn, session.id) == {"ana", "bea"}


def test_replace_thread_content_preserves_statements_when_unchanged(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")
    _ensure_people(conn, "pA")
    msgs = [_msg("id0", "t1", "A", "pA", 0, 0), _msg("id1", "t1", "A", "pA", 1, 1)]
    by_seq = {m.seq: m for m in msgs}

    sessions, preserved = store.replace_thread_content(conn, "t1", msgs, [(0, 1)], by_seq)
    assert preserved == 0
    store.replace_session_statements(
        conn,
        sessions[0].id,
        [_statement(sessions[0].id, source_ids=["id0"], embedding=[0.5, 0.25])],
    )
    store.mark_session_extracted(conn, sessions[0].id, "2024-05-05T00:00:00.000000Z")
    conn.commit()

    # Re-ingesting the same archive must not raise (statements reference
    # sessions that get rebuilt) and must not discard the extraction.
    sessions2, preserved2 = store.replace_thread_content(conn, "t1", msgs, [(0, 1)], by_seq)
    conn.commit()

    assert preserved2 == 1
    kept = store.list_statements(conn, person_id="pA")
    assert len(kept) == 1
    assert kept[0].session_id == sessions2[0].id
    assert kept[0].embedding == pytest.approx([0.5, 0.25])
    assert store.list_sessions(conn)[0].extracted_at == "2024-05-05T00:00:00.000000Z"


def test_replace_thread_content_drops_statements_when_messages_change(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")
    _ensure_people(conn, "pA")
    msgs = [_msg("id0", "t1", "A", "pA", 0, 0), _msg("id1", "t1", "A", "pA", 1, 1)]
    by_seq = {m.seq: m for m in msgs}
    sessions, _ = store.replace_thread_content(conn, "t1", msgs, [(0, 1)], by_seq)
    store.replace_session_statements(
        conn, sessions[0].id, [_statement(sessions[0].id, source_ids=["id0"])]
    )
    store.mark_session_extracted(conn, sessions[0].id, "2024-05-05T00:00:00.000000Z")
    conn.commit()

    # A cited message's content changed, so its id changed -- the statement
    # can no longer be trusted and the session goes back to unextracted.
    edited = [_msg("id0-edited", "t1", "A", "pA", 0, 0, text="different"), msgs[1]]
    _sessions, preserved = store.replace_thread_content(
        conn, "t1", edited, [(0, 1)], {m.seq: m for m in edited}
    )
    conn.commit()

    assert preserved == 0
    assert store.list_statements(conn, person_id="pA") == []
    assert store.list_sessions(conn)[0].extracted_at is None


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


def _statement(session_id, person_id="pA", thread_id="t1", text="hi", source_ids=None, embedding=None):
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
        embedding=embedding,
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


def test_statement_embedding_round_trips_and_defaults_to_none(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")
    _ensure_people(conn, "pA")
    msgs = [_msg("id0", "t1", "A", "pA", 0, 0)]
    store.replace_thread_messages(conn, "t1", msgs)
    by_seq = {m.seq: m for m in msgs}
    [session] = store.replace_thread_sessions(conn, "t1", [(0, 0)], by_seq)
    conn.commit()

    store.replace_session_statements(
        conn, session.id, [_statement(session.id, text="no embedding")]
    )
    conn.commit()
    assert store.list_statements(conn, person_id="pA")[0].embedding is None

    store.replace_session_statements(
        conn, session.id, [_statement(session.id, text="embedded", embedding=[0.1, 0.2, 0.3])]
    )
    conn.commit()
    # Stored as packed float32, so values come back with float32 precision --
    # irrelevant to cosine ranking, which is all embeddings are used for.
    assert store.list_statements(conn, person_id="pA")[0].embedding == pytest.approx(
        [0.1, 0.2, 0.3]
    )


def test_messages_by_ids_preserves_order_and_skips_missing(tmp_path):
    conn = store.connect(tmp_path / "chatmem.db")
    store.upsert_thread(conn, "t1", None, [], "instagram", "2024-01-01T00:00:00Z")
    _ensure_people(conn, "pA")
    msgs = [
        _msg("id0", "t1", "A", "pA", 0, 0, text="first"),
        _msg("id1", "t1", "A", "pA", 1, 1, text="second"),
    ]
    store.replace_thread_messages(conn, "t1", msgs)
    conn.commit()

    loaded = store.messages_by_ids(conn, ["id1", "missing", "id0"])
    assert [m.text for m in loaded] == ["second", "first"]

    assert store.messages_by_ids(conn, []) == []


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
