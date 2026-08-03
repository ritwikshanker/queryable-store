from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC_ARCHIVE = FIXTURES / "synthetic_archive"


@pytest.fixture
def synthetic_archive() -> Path:
    return SYNTHETIC_ARCHIVE


@pytest.fixture
def merge_config_path() -> Path:
    """Config declaring Alex Rivera / Alex R. as one person, for merge tests."""
    return FIXTURES / "merge_config.yaml"
