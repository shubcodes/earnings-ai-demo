# tests/conftest.py
import pytest
import yaml
from pathlib import Path

@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def test_data_dir(tmp_path):
    """Create test data directory structure"""
    (tmp_path / 'audio').mkdir()
    (tmp_path / 'documents').mkdir()
    return tmp_path