import pytest

from chatmem.config import ConfigError, load_config


def _write(tmp_path, body: str):
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_extraction_defaults_when_section_absent(tmp_path):
    cfg = load_config(_write(tmp_path, "target: alex\n"))
    assert cfg.extraction.validation_pass is False
    assert cfg.extraction.dedup_threshold == 0.92


def test_dedup_threshold_is_read_from_config(tmp_path):
    cfg = load_config(_write(tmp_path, "extraction:\n  dedup_threshold: 0.8\n"))
    assert cfg.extraction.dedup_threshold == 0.8


def test_dedup_threshold_zero_disables_dedup(tmp_path):
    cfg = load_config(_write(tmp_path, "extraction:\n  dedup_threshold: 0\n"))
    assert cfg.extraction.dedup_threshold == 0


def test_dedup_threshold_rejects_non_numbers(tmp_path):
    with pytest.raises(ConfigError, match="dedup_threshold"):
        load_config(_write(tmp_path, "extraction:\n  dedup_threshold: 'high'\n"))


def test_dedup_threshold_rejects_values_above_one(tmp_path):
    # Cosine similarity never exceeds 1, so a larger value silently disables
    # dedup rather than doing what the user meant.
    with pytest.raises(ConfigError, match="dedup_threshold"):
        load_config(_write(tmp_path, "extraction:\n  dedup_threshold: 92\n"))
