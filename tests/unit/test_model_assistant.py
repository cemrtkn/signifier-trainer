"""Tests for ModelAssistant class."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from src.config import ModelAssistantConfig
from src.model_assistant import ModelAssistant


class TestModelAssistant:
    """Test suite for ModelAssistant class."""

    @patch("src.model_assistant.AutoTokenizer")
    @patch("src.model_assistant.AutoModelForCausalLM")
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test ModelAssistant initialization."""
        # Mock the model and tokenizer
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()

        config = ModelAssistantConfig(model_path="gpt2")
        assistant = ModelAssistant(config)

        assert assistant.config == config
        assert assistant.model is not None
        assert assistant.tokenizer is not None
        mock_tokenizer.from_pretrained.assert_called_once_with("gpt2")
        mock_model.from_pretrained.assert_called_once_with("gpt2")

    @patch("src.model_assistant.torch.cuda.is_available")
    @patch("src.model_assistant.AutoTokenizer")
    @patch("src.model_assistant.AutoModelForCausalLM")
    def test_device_auto_detection(
        self, mock_model, mock_tokenizer, mock_cuda
    ):
        """Test automatic device detection."""
        mock_cuda.return_value = True
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance

        config = ModelAssistantConfig()  # No device specified
        assistant = ModelAssistant(config)

        assert assistant.device.type == "cuda"

    @patch("src.model_assistant.AutoTokenizer")
    @patch("src.model_assistant.AutoModelForCausalLM")
    def test_tokenizer_fallback(self, mock_model, mock_tokenizer):
        """Test tokenizer fallback to GPT-2 when loading fails."""
        # Mock first call to fail, second to succeed
        mock_tokenizer.from_pretrained.side_effect = [
            Exception("Tokenizer not found"),
            Mock(),
        ]
        mock_model.from_pretrained.return_value = Mock()

        config = ModelAssistantConfig(model_path="invalid-model")
        ModelAssistant(config)

        # Should call tokenizer twice - first with invalid model, then gpt2
        assert mock_tokenizer.from_pretrained.call_count == 2
        mock_tokenizer.from_pretrained.assert_any_call("invalid-model")
        mock_tokenizer.from_pretrained.assert_any_call("gpt2")

    @patch("src.model_assistant.AutoTokenizer")
    @patch("src.model_assistant.AutoModelForCausalLM")
    def test_generate_text(self, mock_model_cls, mock_tokenizer_cls):
        """Test text generation functionality."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock tokenizer behavior
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        mock_tokenizer.decode.return_value = "generated text"
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|endoftext|>"

        # Mock model behavior
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        config = ModelAssistantConfig()
        assistant = ModelAssistant(config)

        result = assistant.generate_text("test prompt")

        assert result == "generated text"
        mock_model.generate.assert_called_once()

    @patch("src.model_assistant.AutoTokenizer")
    @patch("src.model_assistant.AutoModelForCausalLM")
    def test_model_loading_failure(self, mock_model, mock_tokenizer):
        """Test model loading failure raises exception."""
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.side_effect = Exception("Model not found")

        config = ModelAssistantConfig(model_path="nonexistent-model")

        with pytest.raises(Exception, match="Model not found"):
            ModelAssistant(config)


class TestDatasetLoading:
    """Test suite for dataset loading functionality."""

    def test_load_txt_dataset(self):
        """Test loading dataset from text file."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("First example text\n")
            f.write("Second example text\n")
            f.write("Third example text\n")
            temp_path = f.name

        try:
            dataset = ModelAssistant.load_dataset_from_path(temp_path)

            assert len(dataset) == 3
            assert dataset[0]["text"] == "First example text"
            assert dataset[1]["text"] == "Second example text"
        finally:
            Path(temp_path).unlink()

    def test_load_hf_dataset_online(self):
        """Test loading dataset from HuggingFace Hub."""
        # Use rotten_tomatoes as a small, reliable test dataset
        dataset = ModelAssistant.load_dataset_from_path("rotten_tomatoes")

        # Basic validation
        assert len(dataset) > 0
        assert "text" in dataset.column_names

        # Check that we got some actual data
        first_example = dataset[0]
        assert isinstance(first_example["text"], str)
        assert len(first_example["text"]) > 0

    @patch("src.model_assistant.load_dataset")
    def test_dataset_loading_failure(self, mock_load_dataset):
        """Test dataset loading failure raises ValueError."""
        mock_load_dataset.side_effect = Exception("Dataset not found")

        with pytest.raises(ValueError, match="Could not load dataset"):
            ModelAssistant.load_dataset_from_path("nonexistent.json")


@pytest.fixture
def temp_dataset():
    """Create temporary dataset file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write("This is a test sentence.\n")
        f.write("Another example for training.\n")
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()  # Cleanup
