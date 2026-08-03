"""End-to-end extraction against a real LM Studio server.

Skipped by default (pytest.ini's `addopts = -m 'not llm'`). Run explicitly with:

    uv run pytest -m llm tests/test_extract_live.py

against a running LM Studio instance with config.yaml's llm.chat_model set to
a model it actually serves.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from chatmem import store
from chatmem.cli import app
from chatmem.config import load_config

pytestmark = pytest.mark.llm

runner = CliRunner()
CONFIG_PATH = Path("config.yaml")


def test_extract_against_live_lm_studio(tmp_path, synthetic_archive):
    cfg = load_config(CONFIG_PATH)
    if not cfg.llm.chat_model:
        pytest.skip("config.yaml has no llm.chat_model set")

    db_path = tmp_path / "chatmem.db"
    extract_config = tmp_path / "config.yaml"
    extract_config.write_text(
        "target: target\n"
        "identities:\n"
        "  - id: target\n"
        '    display_name: "Alex Rivera"\n'
        '    aliases: ["Alex Rivera", "Alex R."]\n'
        f"llm:\n  chat_model: {cfg.llm.chat_model!r}\n  base_url: {cfg.llm.base_url!r}\n"
    )

    ingest_result = runner.invoke(
        app, ["ingest", str(synthetic_archive), "--config", str(extract_config), "--db", str(db_path)]
    )
    assert ingest_result.exit_code == 0, ingest_result.output

    extract_result = runner.invoke(
        app, ["extract", "--config", str(extract_config), "--db", str(db_path)]
    )
    assert extract_result.exit_code == 0, extract_result.output

    conn = store.connect(db_path)
    statements = store.list_statements(conn, person_id="target")
    # A real model may or may not find self-statements in this tiny fixture
    # archive -- the point of this test is that the round trip doesn't
    # error, not that it finds a specific count.
    for s in statements:
        assert s.text
        assert s.person_id == "target"
