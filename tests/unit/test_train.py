#!/usr/bin/env python3
"""Unit tests for training script functionality."""

import argparse
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from transformers import TrainingArguments

from config import TrainingConfig
from train import (
    execute_training_workflow,
    load_and_process_config,
    main,
    validate_training_setup,
)

# Test constants
TEST_CONFIG_PATH = "test.yaml"
TEST_OUTPUT_DIR = "/tmp/output"
TEST_DATASET_PATH = "/tmp/data.txt"
TEST_DATA_CONTENT = "test data"
TEST_ERROR_MSG = "Training failed"


class TestHelpers:
    """Helper methods for test setup and common operations."""

    @staticmethod
    def create_mock_config(
        dataset_path: str = TEST_DATASET_PATH,
        output_dir: str = TEST_OUTPUT_DIR,
        overwrite_output_dir: bool = True,
    ) -> Mock:
        """Create a mock TrainingConfig with common defaults."""
        config = Mock(spec=TrainingConfig)
        config.dataset_path = dataset_path
        config.output_dir = output_dir
        config.training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
        )
        config.assistant_config = Mock()
        config.assistant_config.model_path = "gpt2"  # Valid model path
        return config

    @staticmethod
    def create_mock_args(
        config: str = TEST_CONFIG_PATH,
        output_dir: str = None,
        dataset_path: str = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> argparse.Namespace:
        """Create mock command-line arguments with common defaults."""
        return argparse.Namespace(
            config=config,
            output_dir=output_dir,
            dataset_path=dataset_path,
            dry_run=dry_run,
            verbose=verbose,
        )

    @staticmethod
    @contextmanager
    def temp_dataset_and_output(create_output_dir: bool = False):
        """Context manager for temporary dataset and output directory setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset file
            dataset_path = Path(tmpdir) / "data.txt"
            dataset_path.write_text(TEST_DATA_CONTENT)

            # Optionally create output directory
            output_dir = Path(tmpdir) / "output"
            if create_output_dir:
                output_dir.mkdir()

            yield str(dataset_path), str(output_dir)


class TestCreateArgumentParser:
    """Tests for create_argument_parser function."""

    def test_creates_parser_with_required_args(self):
        """Test that parser creates all required arguments."""
        from train import create_argument_parser

        parser = create_argument_parser()

        # Test that parser can handle required arguments
        args = parser.parse_args(["--config", "test.yaml"])
        assert args.config == "test.yaml"

    def test_sets_default_values(self):
        """Test that parser sets correct default values."""
        from train import create_argument_parser

        parser = create_argument_parser()
        args = parser.parse_args(["--config", "test.yaml"])

        assert args.dry_run is False
        assert args.verbose is False
        assert args.output_dir is None
        assert args.dataset_path is None


class TestLoadAndProcessConfig:
    """Tests for load_and_process_config function."""

    @patch("train.load_config_from_yaml")
    def test_loads_config_and_applies_overrides(self, mock_load):
        """Test config loading with CLI argument overrides."""
        # Setup mock config
        mock_config = TestHelpers.create_mock_config()
        mock_config.training_args = {}
        mock_load.return_value = mock_config

        # Create args with overrides
        args = TestHelpers.create_mock_args(
            output_dir="/custom/output", dataset_path="/custom/data.txt"
        )

        result = load_and_process_config(args)

        # Verify config loading and overrides
        from config import TrainingConfig as TrainConfigImport

        mock_load.assert_called_once_with(TEST_CONFIG_PATH, TrainConfigImport)
        assert result.output_dir == "/custom/output"
        assert result.dataset_path == "/custom/data.txt"
        assert result.training_args["output_dir"] == "/custom/output"

    @patch("train.load_config_from_yaml")
    def test_handles_no_overrides(self, mock_load):
        """Test config loading without CLI overrides."""
        mock_config = TestHelpers.create_mock_config()
        mock_config.training_args = {}
        mock_load.return_value = mock_config

        args = TestHelpers.create_mock_args()
        result = load_and_process_config(args)

        # Config should be returned as-is without modifications
        assert result == mock_config


class TestValidateTrainingSetup:
    """Tests for validate_training_setup function."""

    def test_validation_passes_with_valid_setup(self):
        """Test validation passes with existing dataset and valid config."""
        with TestHelpers.temp_dataset_and_output() as (
            dataset_path,
            output_dir,
        ):
            config = TestHelpers.create_mock_config(
                dataset_path=dataset_path, output_dir=output_dir
            )

            # Should pass without exceptions
            validate_training_setup(config)

    def test_validation_fails_with_missing_dataset(self):
        """Test validation fails when dataset file doesn't exist."""
        config = TestHelpers.create_mock_config(
            dataset_path="/nonexistent/path.txt"
        )

        with pytest.raises(FileNotFoundError):
            validate_training_setup(config)

    def test_validation_warns_about_existing_output_dir(self, caplog):
        """Test that validation warns about existing output directory."""
        with TestHelpers.temp_dataset_and_output(create_output_dir=True) as (
            dataset_path,
            output_dir,
        ):
            config = TestHelpers.create_mock_config(
                dataset_path=dataset_path,
                output_dir=output_dir,
                overwrite_output_dir=False,
            )

            with caplog.at_level(logging.WARNING):
                # Should pass validation but log warning
                validate_training_setup(config)


class TestExecuteTrainingWorkflow:
    """Tests for execute_training_workflow function."""

    @patch("train.ModelAssistant")
    def test_initializes_assistant_and_runs_training(
        self, mock_assistant_class
    ):
        """Test that ModelAssistant is initialized and training runs."""
        mock_assistant = Mock()
        mock_assistant_class.return_value = mock_assistant
        config = TestHelpers.create_mock_config()

        execute_training_workflow(config)

        # Verify assistant creation and training call
        mock_assistant_class.assert_called_once_with(config.assistant_config)
        mock_assistant.train.assert_called_once_with(config)

    @patch("train.ModelAssistant")
    def test_handles_training_exceptions(self, mock_assistant_class):
        """Test that training exceptions are properly propagated."""
        mock_assistant = Mock()
        mock_assistant.train.side_effect = ValueError(TEST_ERROR_MSG)
        mock_assistant_class.return_value = mock_assistant
        config = TestHelpers.create_mock_config()

        with pytest.raises(ValueError, match=TEST_ERROR_MSG):
            execute_training_workflow(config)


class TestMainFunction:
    """Tests for main function integration."""

    def _setup_main_mocks(self, args: argparse.Namespace = None):
        """Helper to setup common mocks for main function tests."""
        if args is None:
            args = TestHelpers.create_mock_args()

        config = TestHelpers.create_mock_config()
        return args, config

    @patch("train.validate_training_setup")
    @patch("train.execute_training_workflow")
    @patch("train.load_and_process_config")
    @patch("train.create_argument_parser")
    def test_main_dry_run_mode(
        self, mock_parser, mock_load, mock_execute, mock_validate
    ):
        """Test main function in dry run mode."""
        args, config = self._setup_main_mocks()
        args.dry_run = True

        mock_parser.return_value.parse_args.return_value = args
        mock_load.return_value = config

        main()

        # Verify workflow: parse args, load config, validate, but no training
        mock_parser.assert_called_once()
        mock_load.assert_called_once_with(args)
        mock_validate.assert_called_once_with(config)
        mock_execute.assert_not_called()

    @patch("train.validate_training_setup")
    @patch("train.execute_training_workflow")
    @patch("train.load_and_process_config")
    @patch("train.create_argument_parser")
    def test_main_normal_execution(
        self, mock_parser, mock_load, mock_execute, mock_validate
    ):
        """Test main function normal execution path."""
        args, config = self._setup_main_mocks()

        mock_parser.return_value.parse_args.return_value = args
        mock_load.return_value = config

        main()

        # Verify full workflow: parse, load, validate, execute
        mock_parser.assert_called_once()
        mock_load.assert_called_once_with(args)
        mock_validate.assert_called_once_with(config)
        mock_execute.assert_called_once_with(config)

    @patch("train.validate_training_setup")
    @patch("train.load_and_process_config")
    @patch("train.create_argument_parser")
    def test_main_handles_config_errors(
        self, mock_parser, mock_load, mock_validate
    ):
        """Test main function handles configuration errors gracefully."""
        args, _ = self._setup_main_mocks()
        mock_parser.return_value.parse_args.return_value = args
        mock_load.side_effect = FileNotFoundError("Config not found")

        with pytest.raises(SystemExit):
            main()

        mock_validate.assert_not_called()

    @patch("train.execute_training_workflow")
    @patch("train.validate_training_setup")
    @patch("train.load_and_process_config")
    @patch("train.create_argument_parser")
    def test_main_handles_training_errors(
        self, mock_parser, mock_load, mock_validate, mock_execute
    ):
        """Test main function handles training errors gracefully."""
        args, config = self._setup_main_mocks()
        mock_parser.return_value.parse_args.return_value = args
        mock_load.return_value = config
        mock_execute.side_effect = RuntimeError("Training failed")

        with pytest.raises(SystemExit):
            main()

        mock_validate.assert_called_once_with(config)

    @patch("train.validate_training_setup")
    @patch("train.load_and_process_config")
    @patch("train.create_argument_parser")
    def test_main_verbose_logging(self, mock_parser, mock_load, mock_validate):
        """Test that verbose flag enables debug logging."""
        args, config = self._setup_main_mocks()
        args.verbose = True
        args.dry_run = True

        mock_parser.return_value.parse_args.return_value = args
        mock_load.return_value = config

        # Check that logging level changes
        original_level = logging.getLogger().level
        main()
        assert logging.getLogger().level == logging.DEBUG

        # Restore original logging level
        logging.getLogger().setLevel(original_level)
