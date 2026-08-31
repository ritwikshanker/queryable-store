import hashlib
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from chatmem import cli, store
from chatmem.cli import _message_id, app
from chatmem.llm import LLMResponseError

runner = CliRunner()

NO_CONFIG = Path("/nonexistent/config.yaml")  # forces all-defaults, auto-create identities


class _FakeLLM:
    """Stand-in for chatmem.llm.LLMClient: no network, deterministic output."""

    def __init__(self, *args, **kwargs):
        pass

    def extract_statements(self, transcript, target_name):
        # Distinct per session, so extraction dedup doesn't collapse them --
        # dedup itself is covered by tests/test_extract.py.
        return [
            {
                "text": f"{target_name} said something in session {len(transcript)}",
                "message_indices": [0],
            }
        ]

    def validate_statements(self, transcript, target_name, statements):
        return [True for _ in statements]

    def classify_statements(self, statements):
        # Spread across a couple of real topics so the digest renders more
        # than one section; which topic a statement lands in is the model's
        # job, not something these plumbing tests should assert.
        return ["family" if i % 2 == 0 else "work" for i in range(len(statements))]

    def embed(self, texts):
        # A one-hot vector whose position is a stable hash of the text:
        # identical texts match, different texts are orthogonal, and the
        # mapping is the same in every process and every client instance.
        # Ranking itself is covered by tests/test_query.py; these tests only
        # check output plumbing.
        out = []
        for text in texts:
            slot = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 512
            vector = [0.0] * 512
            vector[slot] = 1.0
            out.append(vector)
        return out


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


def test_reingest_after_extract_preserves_statements(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    ingest_args = [
        "ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)
    ]
    runner.invoke(app, ingest_args)
    runner.invoke(app, ["extract", "--config", str(extract_config_path), "--db", str(db_path)])

    conn = store.connect(db_path)
    before = store.list_statements(conn, person_id="target")
    assert before, "extract should have produced statements to preserve"
    conn.close()

    # statements reference sessions that re-ingest rebuilds; this used to
    # fail the sessions foreign key outright.
    second = runner.invoke(app, ingest_args)
    assert second.exit_code == 0, second.output
    assert f"statements preserved: {len(before)}" in second.output

    conn = store.connect(db_path)
    after = store.list_statements(conn, person_id="target")
    assert [s.text for s in after] == [s.text for s in before]
    assert [s.embedding for s in after] == [s.embedding for s in before]


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


def test_extract_command_resumes_instead_of_redoing_work(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)]
    )
    args = ["extract", "--config", str(extract_config_path), "--db", str(db_path)]

    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    conn = store.connect(db_path)
    ids_before = [s.id for s in store.list_statements(conn, person_id="target")]
    n_sessions = len(store.list_sessions(conn))
    assert all(s.extracted_at is not None for s in store.list_sessions(conn))
    conn.close()

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert f"{n_sessions} already extracted" in second.output
    assert "0 processed" in second.output
    # Nothing was re-sent to the model, so the stored rows are untouched.
    conn = store.connect(db_path)
    assert [s.id for s in store.list_statements(conn, person_id="target")] == ids_before
    conn.close()

    forced = runner.invoke(app, args + ["--force"])
    assert forced.exit_code == 0, forced.output
    assert f"{n_sessions} processed" in forced.output


def test_extract_command_keeps_finished_sessions_when_interrupted(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    class _InterruptingLLM(_FakeLLM):
        calls = 0

        def extract_statements(self, transcript, target_name):
            type(self).calls += 1
            if type(self).calls > 1:
                raise KeyboardInterrupt
            return super().extract_statements(transcript, target_name)

    monkeypatch.setattr(cli, "LLMClient", _InterruptingLLM)
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)]
    )

    result = runner.invoke(
        app, ["extract", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 130
    assert "progress was saved" in result.output

    # The session that completed before the interrupt is committed, and a
    # resume run picks up only what's left.
    conn = store.connect(db_path)
    assert len(store.list_statements(conn, person_id="target")) == 1
    done = [s for s in store.list_sessions(conn) if s.extracted_at is not None]
    assert len(done) == 1
    conn.close()

    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    resumed = runner.invoke(
        app, ["extract", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert resumed.exit_code == 0, resumed.output
    assert "1 already extracted" in resumed.output


def test_extract_and_query_refuse_a_changed_embedding_model(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)]
    )
    runner.invoke(app, ["extract", "--config", str(extract_config_path), "--db", str(db_path)])

    # A config pointing at a different embedding model: its vectors are not
    # comparable with what's already stored.
    other_config = tmp_path / "other_model.yaml"
    other_config.write_text(
        extract_config_path.read_text(encoding="utf-8").replace(
            "test-embed-model", "different-embed-model"
        ),
        encoding="utf-8",
    )

    extracted = runner.invoke(
        app, ["extract", "--config", str(other_config), "--db", str(db_path)]
    )
    assert extracted.exit_code == 1
    assert "test-embed-model" in extracted.output

    queried = runner.invoke(
        app, ["query", "anything", "--config", str(other_config), "--db", str(db_path)]
    )
    assert queried.exit_code == 1
    assert "extract --force" in queried.output

    # --force re-embeds everything, so it is allowed through.
    forced = runner.invoke(
        app, ["extract", "--config", str(other_config), "--db", str(db_path), "--force"]
    )
    assert forced.exit_code == 0, forced.output
    conn = store.connect(db_path)
    assert store.get_meta(conn, "embedding_model") == "different-embed-model"


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


def test_extract_command_requires_embedding_model(tmp_path, synthetic_archive):
    # chat_model set, embedding_model not -- must fail on the embedding_model check.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "target: target\n"
        "identities:\n"
        "  - id: target\n"
        '    display_name: "Alex Rivera"\n'
        '    aliases: ["Alex Rivera", "Alex R."]\n'
        "llm:\n  chat_model: test-model\n"
    )
    db_path = tmp_path / "chatmem.db"
    runner.invoke(app, ["ingest", str(synthetic_archive), "--config", str(config_path), "--db", str(db_path)])
    result = runner.invoke(app, ["extract", "--config", str(config_path), "--db", str(db_path)])
    assert result.exit_code == 1
    assert "embedding_model" in result.output


def test_query_command_returns_statement_citation_and_source_message(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)]
    )
    runner.invoke(app, ["extract", "--config", str(extract_config_path), "--db", str(db_path)])

    result = runner.invoke(
        app, ["query", "what does Alex do", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    assert "Alex Rivera said something" in result.output
    assert "thread_alpha" in result.output or "thread_beta" in result.output
    assert ">" in result.output  # a quoted source message line


def test_query_command_requires_embedding_model(tmp_path, synthetic_archive, merge_config_path):
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(merge_config_path), "--db", str(db_path)]
    )
    result = runner.invoke(app, ["query", "anything", "--config", str(merge_config_path), "--db", str(db_path)])
    assert result.exit_code == 1
    assert "embedding_model" in result.output


def test_query_command_reports_when_no_statements_embedded(
    tmp_path, synthetic_archive, extract_config_path
):
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)]
    )
    # No `extract` run -> no statements at all, let alone embedded ones.
    result = runner.invoke(
        app, ["query", "anything", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    assert "chatmem extract" in result.output


def test_parse_when_normalizes_dates_to_stored_timestamp_format():
    from chatmem.cli import _parse_when

    assert _parse_when(None, end=False) is None
    # A bare date covers the whole day, so the two bounds differ.
    assert _parse_when("2024-03-05", end=False) == "2024-03-05T00:00:00.000000Z"
    assert _parse_when("2024-03-05", end=True) == "2024-03-05T23:59:59.999999Z"
    # Full timestamps pass through into the same fixed-width form, which is
    # what makes plain string comparison a correct date filter.
    assert _parse_when("2024-03-05T12:30:00Z", end=False) == "2024-03-05T12:30:00.000000Z"


def test_parse_when_rejects_unreadable_dates():
    from chatmem.cli import _parse_when

    with pytest.raises(ValueError, match="not-a-date"):
        _parse_when("not-a-date", end=False)


def _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch):
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)]
    )
    runner.invoke(app, ["extract", "--config", str(extract_config_path), "--db", str(db_path)])
    return db_path


def test_query_thread_filter_restricts_results(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    result = runner.invoke(
        app,
        ["query", "anything", "--thread", "thread_beta", "--config", str(extract_config_path),
         "--db", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert "thread_beta" in result.output
    assert "thread_alpha" not in result.output


def test_query_date_filters_exclude_out_of_range_statements(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)

    conn = store.connect(db_path)
    spans = [(s.start_ts, s.end_ts) for s in store.list_statements(conn, person_id="target")]
    conn.close()
    assert spans, "need statements to filter"
    earliest_day = min(start for start, _ in spans)[:10]

    # A window that ends before everything was said matches nothing.
    empty = runner.invoke(
        app,
        ["query", "anything", "--until", "1999-01-01", "--config", str(extract_config_path),
         "--db", str(db_path)],
    )
    assert empty.exit_code == 0, empty.output
    assert "No statements fall in that range." in empty.output

    # A window starting on the earliest statement's own day still matches it.
    hit = runner.invoke(
        app,
        ["query", "anything", "--since", earliest_day, "--config", str(extract_config_path),
         "--db", str(db_path)],
    )
    assert hit.exit_code == 0, hit.output
    assert "Alex Rivera said something" in hit.output


def test_query_min_score_can_filter_everything_out(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    # The fake embeds each distinct text as an orthogonal vector, so the
    # question never scores above 0 against any statement.
    result = runner.invoke(
        app,
        ["query", "anything", "--min-score", "0.5", "--config", str(extract_config_path),
         "--db", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert "No matching statements found." in result.output


def test_query_answer_flag_prints_synthesized_answer_above_citations(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    class _AnsweringLLM(_FakeLLM):
        def synthesize_answer(self, question, statements):
            return f"Alex is a person [1]. (from {len(statements)} statements)"

    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    monkeypatch.setattr(cli, "LLMClient", _AnsweringLLM)

    result = runner.invoke(
        app,
        ["query", "who is Alex", "--answer", "--config", str(extract_config_path),
         "--db", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert "Alex is a person [1]." in result.output
    # The numbered list below the answer is what [1] refers to.
    assert "[1] [" in result.output
    assert result.output.index("Alex is a person") < result.output.index("[1] [")


def test_query_answer_requires_chat_model(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    no_chat_config = tmp_path / "no_chat.yaml"
    no_chat_config.write_text(
        extract_config_path.read_text(encoding="utf-8").replace(
            'chat_model: "test-model"', 'chat_model: ""'
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["query", "anything", "--answer", "--config", str(no_chat_config), "--db", str(db_path)],
    )
    assert result.exit_code == 1
    assert "chat_model" in result.output


def test_statements_command_lists_and_filters(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)

    listed = runner.invoke(
        app, ["statements", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert listed.exit_code == 0, listed.output
    assert "Alex Rivera said something" in listed.output

    filtered = runner.invoke(
        app,
        ["statements", "--thread", "thread_beta", "--config", str(extract_config_path),
         "--db", str(db_path)],
    )
    assert filtered.exit_code == 0, filtered.output
    assert "thread_alpha" not in filtered.output

    by_person = runner.invoke(
        app,
        ["statements", "--person", "Alex Rivera", "--config", str(extract_config_path),
         "--db", str(db_path)],
    )
    assert by_person.exit_code == 0, by_person.output
    assert "target" in by_person.output


def test_statements_command_reports_unknown_person(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    result = runner.invoke(
        app,
        ["statements", "--person", "Nobody", "--config", str(extract_config_path),
         "--db", str(db_path)],
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_export_json_round_trips(tmp_path, synthetic_archive, extract_config_path, monkeypatch):
    import json as json_module

    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    result = runner.invoke(
        app, ["export", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output

    records = json_module.loads(result.output)
    assert records, "expected exported statements"
    assert {"id", "person_id", "text", "source_message_ids"} <= set(records[0])
    assert "embedding" not in records[0]  # too big to be useful in an export
    assert isinstance(records[0]["source_message_ids"], list)


def test_export_csv_has_a_header_and_parses_back(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    import csv as csv_module
    import io as io_module

    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    result = runner.invoke(
        app, ["export", "--format", "csv", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output

    rows = list(csv_module.DictReader(io_module.StringIO(result.output)))
    assert rows
    assert rows[0]["person_id"] == "target"
    assert "said something" in rows[0]["text"]


def test_export_writes_to_a_file_when_out_given(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    out_path = tmp_path / "out" / "statements.json"
    result = runner.invoke(
        app,
        ["export", "--out", str(out_path), "--config", str(extract_config_path),
         "--db", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert "wrote" in result.output


def test_export_rejects_unknown_format(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    result = runner.invoke(
        app,
        ["export", "--format", "xml", "--config", str(extract_config_path), "--db", str(db_path)],
    )
    assert result.exit_code == 1
    assert "unknown --format" in result.output


def test_ingest_reads_a_whatsapp_export(tmp_path):
    chat = tmp_path / "WhatsApp Chat with Dana.txt"
    chat.write_text(
        "12/05/2023, 9:41 AM - Messages and calls are end-to-end encrypted.\n"
        "12/05/2023, 9:41 AM - Dana: hey there\n"
        "12/05/2023, 9:42 AM - Alex: hi! I moved to Berlin last month\n"
        "12/05/2023, 9:43 AM - Dana: <Media omitted>\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "chatmem.db"

    result = runner.invoke(
        app, ["ingest", str(chat), "--config", str(NO_CONFIG), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    assert "threads:  1" in result.output
    assert "messages: 3 kept" in result.output
    assert "system=1" in result.output

    conn = store.connect(db_path)
    senders = {r["sender"] for r in conn.execute("SELECT DISTINCT sender FROM messages")}
    assert senders == {"Dana", "Alex"}


def test_query_refuses_mixed_embedding_sizes(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)

    # Simulate an `extract --force` that was interrupted after a model change:
    # one statement carries a vector of a different width.
    conn = store.connect(db_path)
    rows = store.list_statements(conn, person_id="target")
    assert len(rows) > 1
    conn.execute(
        "UPDATE statements SET embedding = ? WHERE id = ?",
        (store._pack_embedding([0.1, 0.2, 0.3]), rows[0].id),
    )
    conn.commit()
    conn.close()

    result = runner.invoke(
        app, ["query", "anything", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 1
    assert "mixed embedding sizes" in result.output


# --- digest --------------------------------------------------------------


class _FailingClassifier(_FakeLLM):
    """Classifies nothing: every batch raises, as a flaky server would."""

    def classify_statements(self, statements):
        raise LLMResponseError("boom")


def test_digest_command_classifies_and_writes_every_statement(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    out = tmp_path / "MEMORIES.md"

    result = runner.invoke(
        app, ["digest", "--out", str(out), "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output

    conn = store.connect(db_path)
    stored = store.list_statements(conn, person_id="target")
    assert stored and all(s.topic is not None for s in stored)

    markdown = out.read_text(encoding="utf-8")
    bullets = [line for line in markdown.splitlines() if line.startswith("- **")]
    assert len(bullets) == len(stored)
    assert f"wrote {len(stored)} memories" in result.output


def test_digest_command_only_classifies_statements_it_has_not_seen(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    """Re-running after a new extract must cost one batch for the new rows,
    not a full re-classification of the archive."""
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    out = tmp_path / "MEMORIES.md"
    args = ["digest", "--out", str(out), "--config", str(extract_config_path), "--db", str(db_path)]
    runner.invoke(app, args)

    seen: list[int] = []

    class _CountingLLM(_FakeLLM):
        def classify_statements(self, statements):
            seen.append(len(statements))
            return super().classify_statements(statements)

    monkeypatch.setattr(cli, "LLMClient", _CountingLLM)
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert seen == []

    third = runner.invoke(app, args + ["--reclassify"])
    assert third.exit_code == 0, third.output
    assert sum(seen) == len(store.list_statements(store.connect(db_path), person_id="target"))


def test_digest_command_batches_by_batch_size(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    sizes: list[int] = []

    class _CountingLLM(_FakeLLM):
        def classify_statements(self, statements):
            sizes.append(len(statements))
            return super().classify_statements(statements)

    monkeypatch.setattr(cli, "LLMClient", _CountingLLM)
    result = runner.invoke(
        app,
        ["digest", "--out", str(tmp_path / "M.md"), "--batch-size", "2",
         "--config", str(extract_config_path), "--db", str(db_path)],
    )
    assert result.exit_code == 0, result.output
    assert sizes and all(n <= 2 for n in sizes)
    assert sum(sizes) == len(store.list_statements(store.connect(db_path), person_id="target"))


def test_digest_command_still_writes_everything_when_classification_fails(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    """A dead classifier degrades the grouping, not the completeness -- and
    leaves the rows untagged so the next run retries them."""
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    monkeypatch.setattr(cli, "LLMClient", _FailingClassifier)
    out = tmp_path / "MEMORIES.md"

    result = runner.invoke(
        app, ["digest", "--out", str(out), "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    assert "left unclassified" in result.output

    stored = store.list_statements(store.connect(db_path), person_id="target")
    markdown = out.read_text(encoding="utf-8")
    assert "## Unclassified" in markdown
    assert len([line for line in markdown.splitlines() if line.startswith("- **")]) == len(stored)
    assert all(s.topic is None for s in stored)


def test_digest_command_quotes_source_messages_only_with_the_flag(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    plain = tmp_path / "plain.md"
    quoted = tmp_path / "quoted.md"

    base = ["--config", str(extract_config_path), "--db", str(db_path)]
    runner.invoke(app, ["digest", "--out", str(plain)] + base)
    result = runner.invoke(app, ["digest", "--out", str(quoted), "--quotes"] + base)
    assert result.exit_code == 0, result.output

    assert "  > " not in plain.read_text(encoding="utf-8")
    # Whoever sent the cited message -- the quote is of the source, which is
    # not necessarily the target's own line.
    assert "  > Sam Chen: hey you around?" in quoted.read_text(encoding="utf-8")


def test_digest_command_errors_when_nothing_has_been_extracted(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config_path), "--db", str(db_path)]
    )
    result = runner.invoke(
        app, ["digest", "--out", str(tmp_path / "M.md"), "--config", str(extract_config_path),
              "--db", str(db_path)]
    )
    assert result.exit_code == 1
    assert "Run `chatmem extract` first" in result.output


def test_digest_command_rejects_a_zero_batch_size(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    result = runner.invoke(
        app,
        ["digest", "--out", str(tmp_path / "M.md"), "--batch-size", "0",
         "--config", str(extract_config_path), "--db", str(db_path)],
    )
    assert result.exit_code == 1
    assert "--batch-size" in result.output


# --- incremental re-ingest -------------------------------------------------


def test_reingest_with_a_prepended_older_file_preserves_extraction(
    tmp_path, extract_config_path, monkeypatch
):
    """Instagram splits an export so message_2.json holds OLDER messages. A
    fuller re-export therefore prepends history, shifting every seq in the
    thread. Extraction of the untouched later conversation must survive that,
    or every re-export costs a full re-extract of the whole archive."""
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    inbox = tmp_path / "inbox" / "thread_alpha"
    inbox.mkdir(parents=True)
    src = Path("tests/fixtures/synthetic_archive/thread_alpha")
    shutil.copy(src / "message_1.json", inbox)

    args = ["--config", str(extract_config_path), "--db", str(db_path)]
    runner.invoke(app, ["ingest", str(tmp_path / "inbox")] + args)
    runner.invoke(app, ["extract"] + args)

    conn = store.connect(db_path)
    before = store.list_statements(conn, thread_id="thread_alpha")
    assert before, "extract should have produced statements to preserve"
    conn.close()

    # The fuller export arrives, adding strictly older messages.
    shutil.copy(src / "message_2.json", inbox)
    result = runner.invoke(app, ["ingest", str(tmp_path / "inbox")] + args)
    assert result.exit_code == 0, result.output

    conn = store.connect(db_path)
    after = store.list_statements(conn, thread_id="thread_alpha")
    sessions = store.list_sessions(conn, "thread_alpha")

    # Sessions the prepend did not touch keep their extraction. The session at
    # the join legitimately changes -- the prepended history is close enough in
    # time to merge into it -- so it is correctly queued for re-extraction, and
    # a full preserve is not the contract here.
    assert after, "seq shifts alone must not discard extraction"
    assert {s.text for s in after} <= {s.text for s in before}
    assert "statements preserved:" in result.output
    assert any(s.extracted_at is not None for s in sessions)
    assert any(s.extracted_at is None for s in sessions)


def test_message_ids_are_stable_when_older_history_is_prepended(
    tmp_path, extract_config_path, monkeypatch
):
    """The id is what statements cite, so it must not depend on a message's
    position in the thread -- only on the message itself."""
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    inbox = tmp_path / "inbox" / "thread_alpha"
    inbox.mkdir(parents=True)
    src = Path("tests/fixtures/synthetic_archive/thread_alpha")
    shutil.copy(src / "message_1.json", inbox)
    args = ["--config", str(extract_config_path), "--db", str(db_path)]

    runner.invoke(app, ["ingest", str(tmp_path / "inbox")] + args)
    conn = store.connect(db_path)
    before = {m.id: m.timestamp_ms for m in store.thread_messages(conn, "thread_alpha")}
    conn.close()

    shutil.copy(src / "message_2.json", inbox)
    runner.invoke(app, ["ingest", str(tmp_path / "inbox")] + args)
    conn = store.connect(db_path)
    after = {m.id: m.timestamp_ms for m in store.thread_messages(conn, "thread_alpha")}

    # Every previously-seen message kept its id, despite its seq changing.
    assert set(before) <= set(after)
    # And the seqs really did shift, or this test proves nothing.
    seqs = {m.id: m.seq for m in store.thread_messages(conn, "thread_alpha")}
    assert any(seqs[mid] != i for i, mid in enumerate(before))


# --- reembed ---------------------------------------------------------------


class _WideEmbedLLM(_FakeLLM):
    """A "better" model: same texts, different (wider) vector space."""

    def embed(self, texts):
        return [[0.25] * 1024 for _ in texts]


def test_reembed_replaces_vectors_without_re_running_extraction(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    """The whole point: swapping the embedding model must not cost the chat
    pass again, and must not renumber statements that MEMORIES.md cites."""
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    conn = store.connect(db_path)
    before = store.list_statements(conn)
    assert before and all(s.embedding is not None for s in before)
    conn.close()

    calls = []

    class _NoChatLLM(_WideEmbedLLM):
        def extract_statements(self, transcript, target_name):
            calls.append("chat")
            raise AssertionError("reembed must not call the chat model")

    monkeypatch.setattr(cli, "LLMClient", _NoChatLLM)
    monkeypatch.setattr(
        cli, "_load_cfg", _cfg_with_embed_model(extract_config_path, "better-embed-model")
    )
    result = runner.invoke(app, ["reembed", "--db", str(db_path)])
    assert result.exit_code == 0, result.output
    assert calls == []

    conn = store.connect(db_path)
    after = store.list_statements(conn)
    assert [s.id for s in after] == [s.id for s in before]
    assert [s.text for s in after] == [s.text for s in before]
    assert [s.topic for s in after] == [s.topic for s in before]
    assert all(len(s.embedding) == 1024 for s in after)
    assert store.get_meta(conn, "embedding_model") == "better-embed-model"
    assert store.get_meta(conn, "embedding_dim") == "1024"


def _cfg_with_embed_model(config_path, model):
    from chatmem.config import LLMConfig, load_config

    def loader(cp, db=None, target=None):
        cfg = load_config(config_path).with_overrides(target=target, db_path=db)
        llm = cfg.llm
        return type(cfg)(
            db_path=cfg.db_path,
            target=cfg.target,
            identities=cfg.identities,
            llm=LLMConfig(
                base_url=llm.base_url,
                api_key=llm.api_key,
                chat_model=llm.chat_model,
                embedding_model=model,
                timeout_seconds=llm.timeout_seconds,
                max_retries=llm.max_retries,
                reasoning_effort=llm.reasoning_effort,
            ),
            sessionize=cfg.sessionize,
            extraction=cfg.extraction,
        )

    return loader


def test_reembed_is_a_no_op_when_the_model_already_matches(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    db_path = _seed_for_query(tmp_path, synthetic_archive, extract_config_path, monkeypatch)
    result = runner.invoke(
        app, ["reembed", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 0, result.output
    assert "already embedded" in result.output


def test_reembed_errors_when_nothing_has_been_extracted(tmp_path, extract_config_path):
    db_path = tmp_path / "empty.db"
    store.connect(db_path).close()
    result = runner.invoke(
        app, ["reembed", "--config", str(extract_config_path), "--db", str(db_path)]
    )
    assert result.exit_code == 1
    assert "Run `chatmem extract` first" in result.output


def test_extract_force_rebuilds_instead_of_deduping_against_its_own_output(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    """--force deletes and rebuilds each session's statements. Seeding the
    dedup set from those same statements made every re-extracted statement a
    near-duplicate of the copy it was replacing, so --force emptied the store
    instead of rebuilding it."""
    monkeypatch.setattr(cli, "LLMClient", _FakeLLM)
    db_path = tmp_path / "chatmem.db"
    args = ["--config", str(extract_config_path), "--db", str(db_path)]
    runner.invoke(app, ["ingest", str(synthetic_archive)] + args)
    runner.invoke(app, ["extract"] + args)

    before = store.list_statements(store.connect(db_path), person_id="target")
    assert before, "extract should have produced statements to rebuild"

    result = runner.invoke(app, ["extract", "--force"] + args)
    assert result.exit_code == 0, result.output
    after = store.list_statements(store.connect(db_path), person_id="target")
    assert [s.text for s in after] == [s.text for s in before]
    assert "0 extracted" not in result.output


def test_extract_still_dedups_against_sessions_it_is_not_rebuilding(
    tmp_path, synthetic_archive, extract_config_path, monkeypatch
):
    """The narrowed dedup set must not disable dedup itself: a resumed run
    still compares new statements against the sessions already done."""
    monkeypatch.setattr(cli, "LLMClient", _SameStatementLLM)
    db_path = tmp_path / "chatmem.db"
    args = ["--config", str(extract_config_path), "--db", str(db_path)]
    runner.invoke(app, ["ingest", str(synthetic_archive)] + args)
    result = runner.invoke(app, ["extract"] + args)
    assert result.exit_code == 0, result.output

    # Every session yields the identical statement, so exactly one survives.
    kept = store.list_statements(store.connect(db_path), person_id="target")
    assert len(kept) == 1, f"expected dedup to collapse these, got {len(kept)}"


class _SameStatementLLM(_FakeLLM):
    """Returns the same statement for every session, so dedup must collapse."""

    def extract_statements(self, transcript, target_name):
        return [{"text": "the very same fact every time", "message_indices": [0]}]
