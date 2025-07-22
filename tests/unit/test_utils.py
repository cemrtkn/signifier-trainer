"""Tests for utils module functionality."""

import os
import tempfile

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from src.utils import load_config_from_yaml


class SampleConfig(BaseModel):
    """Sample Pydantic model for testing configuration loading."""

    name: str
    port: int
    debug: bool = False


class TestLoadConfigFromYaml:
    """Test suite for load_config_from_yaml function."""

    def test_load_valid_config(self):
        """Test loading a valid YAML configuration file."""
        config_data = {"name": "test_app", "port": 8080, "debug": True}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp_file:
            yaml.dump(config_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            config = load_config_from_yaml(tmp_path, SampleConfig)

            assert config.name == "test_app"
            assert config.port == 8080
            assert config.debug is True
            assert isinstance(config, SampleConfig)
        finally:
            os.unlink(tmp_path)

    def test_validation_error(self):
        """Test ValidationError when required field is missing."""
        config_data = {"name": "test_app"}  # missing required 'port'

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp_file:
            yaml.dump(config_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            with pytest.raises(ValidationError):
                load_config_from_yaml(tmp_path, SampleConfig)
        finally:
            os.unlink(tmp_path)
