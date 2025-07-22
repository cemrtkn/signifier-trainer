"""Integration tests for training functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from transformers import TrainingArguments

from src.config import ModelAssistantConfig, TrainingConfig
from src.model_assistant import ModelAssistant


class TestTrainingIntegration:
    """Test suite for training workflow integration."""

    @patch("src.model_assistant.Trainer")
    def test_gpt2_training_with_sample_data(self, mock_trainer_class):
        """Test GPT-2 training setup with sample_data.txt file."""
        # Setup mock trainer instance
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        # Create training config for GPT-2 with sample data
        config = TrainingConfig(
            dataset_path="examples/sample_data.txt",
            output_dir="./test_output",
            assistant_config=ModelAssistantConfig(
                model_path="gpt2",
                generation_params={"temperature": 0.7, "max_new_tokens": 20},
            ),
            training_args={
                "output_dir": "./test_output",
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "logging_steps": 10,
                "eval_strategy": "no",  # Disable eval for simplicity
                "save_steps": 500,
                "overwrite_output_dir": True,
                "dataloader_drop_last": True,
                "push_to_hub": False,
            },
            max_length=128,  # Smaller for test
            train_test_split=0.8,
        )

        # Initialize ModelAssistant
        assistant = ModelAssistant(config.assistant_config)

        # Verify model and tokenizer are loaded
        assert assistant.model is not None
        assert assistant.tokenizer is not None
        assert assistant.tokenizer.pad_token is not None

        # Test dataset loading
        datasets = assistant.prepare_dataset_for_training(
            config.dataset_path, config.max_length, config.train_test_split
        )

        # Verify dataset preparation
        assert "train" in datasets
        assert "eval" in datasets
        assert len(datasets["train"]) > 0
        assert len(datasets["eval"]) > 0

        with tempfile.TemporaryDirectory() as temp_dir:
            # Update config with temp directory
            config.output_dir = temp_dir
            # Create new TrainingArguments with updated output_dir
            config.training_args = TrainingArguments(
                output_dir=temp_dir,
                num_train_epochs=config.training_args.num_train_epochs,
                per_device_train_batch_size=config.training_args.per_device_train_batch_size,  # noqa: E501
                per_device_eval_batch_size=config.training_args.per_device_eval_batch_size,  # noqa: E501
                logging_steps=config.training_args.logging_steps,
                eval_strategy=config.training_args.eval_strategy,
                save_steps=config.training_args.save_steps,
                overwrite_output_dir=config.training_args.overwrite_output_dir,
                dataloader_drop_last=config.training_args.dataloader_drop_last,
                push_to_hub=config.training_args.push_to_hub,
            )

            # Start training (with mocked trainer)
            assistant.train(config)

            # Verify trainer was initialized and called
            mock_trainer_class.assert_called_once()
            mock_trainer.train.assert_called_once()
            mock_trainer.save_model.assert_called_once_with(temp_dir)

            # Verify training config was saved
            config_file = Path(temp_dir) / "training_config.json"
            assert config_file.exists()

    def test_gpt2_text_generation(self):
        """Test GPT-2 text generation functionality."""
        # Create ModelAssistant with GPT-2
        config = ModelAssistantConfig(model_path="gpt2")
        assistant = ModelAssistant(config)

        # Test text generation functionality
        prompt = "Once upon a time"
        generated_text = assistant.generate_text(
            prompt,
            generation_params={
                "max_new_tokens": 10,
                "temperature": 0.7,
                "do_sample": False,  # Deterministic for testing
            },
        )

        # Verify generation worked
        assert isinstance(generated_text, str)
        assert len(generated_text.strip()) > 0
