from pathlib import Path

from typer.testing import CliRunner

from chatmem import cli, store
from chatmem.cli import _message_id, app

runner = CliRunner()

NO_CONFIG = Path("/nonexistent/config.yaml")  # forces all-defaults, auto-create identities


class _FakeLLM:
    """Stand-in for chatmem.llm.LLMClient: no network, deterministic output."""

    def __init__(self, *args, **kwargs):
        pass

    def extract_statements(self, transcript, target_name):
        return [{"text": f"{target_name} said something", "message_indices": [0]}]

    def validate_statements(self, transcript, target_name, statements):
        return [True for _ in statements]


def test_message_id_is_deterministic_and_distinguishes_ordinal():
    a = _message_id("t1", 100, "Alice", "hi", None, 0)
    b = _message_id("t1", 100, "Alice", "hi", None, 0)
    c = _message_id("t1", 100, "Alice", "hi", None, 1)
    assert a == b
    assert a != c


def test_ingest_reports_counts_and_new_sender_warning(tmp_path, synthetic_archive):
    db_path = tmp_path / "chatmem.db"
    result = runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(NO_CONFIG), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    assert "threads:  2" in result.output
    assert "messages: 28 kept" in result.output  # 23 (alpha) + 5 (beta)
    assert "sessions:" in result.output
    assert "Warning:" in result.output  # no identities declared -> everyone auto-created


def test_ingest_with_merge_config_merges_person_across_threads(tmp_path, synthetic_archive, merge_config_path):
    db_path = tmp_path / "chatmem.db"
    result = runner.invoke(
        app,
        ["ingest", str(synthetic_archive), "--config", str(merge_config_path), "--db", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert "target resolved to person_id='target'" in result.output

    conn = store.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT thread_id FROM messages WHERE person_id = 'target' ORDER BY thread_id"
    ).fetchall()
    assert [r["thread_id"] for r in rows] == ["thread_alpha", "thread_beta"]

    # "Alex Rivera" and "Alex R." both resolve to the same person.
    senders = conn.execute(
        "SELECT DISTINCT sender FROM messages WHERE person_id = 'target' ORDER BY sender"
    ).fetchall()
    assert {r["sender"] for r in senders} == {"Alex Rivera", "Alex R."}


def test_ingest_is_idempotent(tmp_path, synthetic_archive, merge_config_path):
    db_path = tmp_path / "chatmem.db"
    args = ["ingest", str(synthetic_archive), "--config", str(merge_config_path), "--db", str(db_path)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    conn = store.connect(db_path)
    count_after_first = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    count_after_second = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]

    assert count_after_first == count_after_second


def test_identities_command_lists_people(tmp_path, synthetic_archive, merge_config_path):
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(merge_config_path), "--db", str(db_path)]
    )
    result = runner.invoke(app, ["identities", "--config", str(merge_config_path), "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "target" in result.output
    assert "other" in result.output


def test_stats_command_runs(tmp_path, synthetic_archive, merge_config_path):
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(merge_config_path), "--db", str(db_path)]
    )
    result = runner.invoke(app, ["stats", "--config", str(merge_config_path), "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "thread_alpha" in result.output
    assert "thread_beta" in result.output


def test_sessions_command_runs(tmp_path, synthetic_archive, merge_config_path):
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(merge_config_path), "--db", str(db_path)]
    )
    result = runner.invoke(app, ["sessions", "--config", str(merge_config_path), "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "thread_alpha" in result.output


def test_relink_after_config_change(tmp_path, synthetic_archive):
    db_path = tmp_path / "chatmem.db"
    # Ingest with no identities declared -> Alex Rivera / Alex R. stay separate.
    runner.invoke(app, ["ingest", str(synthetic_archive), "--config", str(NO_CONFIG), "--db", str(db_path)])

    conn = store.connect(db_path)
    before = {
        r["person_id"]
        for r in conn.execute("SELECT DISTINCT person_id FROM messages WHERE sender IN ('Alex Rivera', 'Alex R.')")
    }
    assert len(before) == 2  # not yet merged

    result = runner.invoke(
        app, ["relink", "--config", str((Path(__file__).parent / "fixtures" / "merge_config.yaml")), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output

    after = {
        r["person_id"]
        for r in conn.execute("SELECT DISTINCT person_id FROM messages WHERE sender IN ('Alex Rivera', 'Alex R.')")
    }
    assert after == {"target"}


def test_extract_command_stores_statements_for_target_sessions_only(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)]
    )

    result = runner.invoke(app, ["extract", "--config", str(extract_config_path), "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "statements:" in result.output

    conn = store.connect(db_path)
    rows = conn.execute("SELECT person_id, text FROM statements").fetchall()
    assert len(rows) > 0
    assert all(r["person_id"] == "target" for r in rows)
    assert all("Alex Rivera" in r["text"] for r in rows)


def test_extract_command_is_idempotent(tmp_path, synthetic_archive, extract_config_path, monkeypatch):
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)]
    )
    args = ["extract", "--config", str(extract_config_path), "--db", str(db_path)]
    runner.invoke(app, args)
    first_count = store.connect(db_path).execute("SELECT COUNT(*) AS n FROM statements").fetchone()["n"]

    runner.invoke(app, args)
    second_count = store.connect(db_path).execute("SELECT COUNT(*) AS n FROM statements").fetchone()["n"]
    assert first_count == second_count


def test_extract_command_requires_target(tmp_path, synthetic_archive):
    db_path = tmp_path / "chatmem.db"
    result = runner.invoke(app, ["extract", "--config", str(NO_CONFIG), "--db", str(db_path)])
    assert result.exit_code == 1
    assert "no target set" in result.output


def test_extract_command_requires_chat_model(tmp_path, synthetic_archive, merge_config_path):
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(merge_config_path), "--db", str(db_path)]
    )
    result = runner.invoke(app, ["extract", "--config", str(merge_config_path), "--db", str(db_path)])
    assert result.exit_code == 1
    assert "chat_model" in result.output
