"""Tests for configuration classes."""

import pytest
from pydantic import ValidationError

from config import InferenceConfig, ModelAssistantConfig, TrainingConfig
from utils import load_config_from_yaml


class TestModelAssistantConfig:
    """Test suite for ModelAssistantConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelAssistantConfig()

        assert config.model_path == "gpt2"
        assert "do_sample" in config.generation_params
        assert config.generation_params["temperature"] == 0.7

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_params = {"temperature": 0.5, "max_new_tokens": 100}
        config = ModelAssistantConfig(
            model_path="microsoft/DialoGPT-medium",
            generation_params=custom_params,
        )

        assert config.model_path == "microsoft/DialoGPT-medium"
        assert config.generation_params["temperature"] == 0.5


class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_minimal_config(self):
        """Test minimal required configuration."""
        config = TrainingConfig(
            dataset_path="test_dataset.txt", output_dir="./output"
        )

        assert config.dataset_path == "test_dataset.txt"
        assert config.output_dir == "./output"
        assert config.max_length == 512
        assert config.train_test_split == 0.8

    def test_validation_error(self):
        """Test ValidationError when required fields are missing."""
        with pytest.raises(ValidationError):
            TrainingConfig()  # Missing required dataset_path and output_dir

        with pytest.raises(ValidationError):
            TrainingConfig(dataset_path="test.txt")  # Missing output_dir

    def test_load_example_config_yaml(self):
        """Test loading the example config from examples directory."""
        config = load_config_from_yaml(
            "examples/example_config.yaml", TrainingConfig
        )

        # Verify basic structure
        assert config.dataset_path == "examples/sample_data.txt"
        assert (
            config.output_dir == "trained_models/<<group_name>>/<<job_name>>"
        )
        assert config.max_length == 128
        assert config.train_test_split == 0.7

        # Verify assistant config
        assert config.assistant_config.model_path == "gpt2"
        assert config.assistant_config.generation_params["temperature"] == 0.8
        assert (
            config.assistant_config.generation_params["max_new_tokens"] == 30
        )

        # Verify training args
        assert config.training_args.num_train_epochs == 1
        assert config.training_args.per_device_train_batch_size == 2
        assert (
            config.training_args.output_dir
            == "trained_models/<<group_name>>/<<job_name>>"
        )


class TestInferenceConfig:
    """Test suite for InferenceConfig."""

    def test_minimal_config(self):
        """Test minimal required configuration."""
        config = InferenceConfig(model_path="./trained_model")

        assert config.model_path == "./trained_model"
        assert config.batch_size == 1
        assert "do_sample" in config.generation_params
        assert config.generation_params["temperature"] == 0.7

    def test_validation_error(self):
        """Test ValidationError when required field is missing."""
        with pytest.raises(ValidationError):
            InferenceConfig()  # Missing required model_path
