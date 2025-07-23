"""Tests for run_slurm module functionality."""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from run_slurm import (
    ConfigProcessor,
    FileManager,
    ResourceCalculator,
    SlurmJobError,
    SlurmJobManager,
    SlurmSubmitter,
)


class TestConfigProcessor:
    """Test suite for ConfigProcessor class."""

    @pytest.fixture
    def config_processor(self):
        """Create ConfigProcessor instance for testing."""
        return ConfigProcessor()

    def test_generate_job_id(self, config_processor):
        """Test job ID generation with timestamp format."""
        job_id = config_processor.generate_job_id()
        assert len(job_id) == 20  # YYYY_MM_DD__HH_MM_SS format
        assert job_id.count("_") == 6

    @patch("run_slurm.datetime")
    def test_generate_job_id_unique(self, mock_datetime, config_processor):
        """Test that consecutive job IDs are unique."""
        mock_datetime.now.side_effect = [
            MagicMock(strftime=lambda fmt: "2024_01_01__12_00_00"),
            MagicMock(strftime=lambda fmt: "2024_01_01__12_00_01"),
        ]
        id1 = config_processor.generate_job_id()
        id2 = config_processor.generate_job_id()
        assert id1 != id2

    def test_parse_dynamic_args_single_arg(self, config_processor):
        """Test parsing single dynamic argument."""
        unknown_args = ["--model_name"]
        argv = ["script.py", "--model_name", "gpt2"]
        result = config_processor.parse_dynamic_args(unknown_args, argv)
        assert result == {"model_name": "gpt2"}

    def test_parse_dynamic_args_multiple(self, config_processor):
        """Test parsing multiple dynamic arguments."""
        unknown_args = ["--epochs", "--batch_size"]
        argv = ["script.py", "--epochs", "10", "--batch_size", "32"]
        result = config_processor.parse_dynamic_args(unknown_args, argv)
        expected = {"epochs": "10", "batch_size": "32"}
        assert result == expected

    def test_parse_dynamic_args_missing_value(self, config_processor):
        """Test parsing dynamic args when argument is missing from argv."""
        unknown_args = ["--missing_arg", "--present_arg"]
        argv = ["script.py", "--present_arg", "value"]
        result = config_processor.parse_dynamic_args(unknown_args, argv)
        # Only the present arg should be parsed
        assert result == {"present_arg": "value"}

    def test_parse_dynamic_args_empty_unknown(self, config_processor):
        """Test parsing with empty unknown_args."""
        unknown_args = []
        argv = ["script.py", "--some_arg", "value"]
        result = config_processor.parse_dynamic_args(unknown_args, argv)
        assert result == {}

    def test_load_config_simple(self, config_processor):
        """Test loading simple YAML configuration."""
        config_data = {"model": "gpt2", "epochs": 5}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp_file:
            yaml.dump(config_data, tmp_file)
            tmp_file.flush()

            result = config_processor.load_and_merge_configs(tmp_file.name)
            assert result == config_data

        os.unlink(tmp_file.name)

    def test_load_config_with_include(self, config_processor):
        """Test loading configuration with includes."""
        # Create included config
        included_data = {"learning_rate": 0.001, "optimizer": "adam"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as included_file:
            yaml.dump(included_data, included_file)
            included_file.flush()

            # Create main config with include
            main_data = {
                "model": "gpt2",
                "training": {"__include": [included_file.name], "epochs": 10},
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as main_file:
                yaml.dump(main_data, main_file)
                main_file.flush()

                result = config_processor.load_and_merge_configs(
                    main_file.name
                )
                # Includes are merged at top level, not into the section
                assert result["model"] == "gpt2"
                assert result["learning_rate"] == 0.001  # Top level
                assert result["optimizer"] == "adam"  # Top level
                assert result["training"]["epochs"] == 10

        os.unlink(included_file.name)
        os.unlink(main_file.name)

    def test_load_config_with_nested_merge(self, config_processor):
        """Test deep merging with nested dictionaries."""
        # Create included config with nested structure
        included_data = {
            "optimizer": {"lr": 0.001, "momentum": 0.9},
            "scheduler": {"type": "cosine"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as included_file:
            yaml.dump(included_data, included_file)
            included_file.flush()

            # Create main config with overlapping nested structure
            main_data = {
                "model": "gpt2",
                "optimizer": {
                    "lr": 0.002,
                    "weight_decay": 0.01,
                },  # Should deep merge
                "training": {"__include": [included_file.name], "epochs": 100},
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as main_file:
                yaml.dump(main_data, main_file)
                main_file.flush()

                result = config_processor.load_and_merge_configs(
                    main_file.name
                )
                # Check nested merge happened correctly
                # included overrides main
                assert result["optimizer"]["lr"] == 0.001  # From included
                assert result["optimizer"]["momentum"] == 0.9  # From included
                assert (
                    result["optimizer"]["weight_decay"] == 0.01
                )  # From main, preserved
                assert result["scheduler"]["type"] == "cosine"  # From included
                assert result["training"]["epochs"] == 100  # From main

        os.unlink(included_file.name)
        os.unlink(main_file.name)

    def test_load_config_file_not_found(self, config_processor):
        """Test error handling for missing config file."""
        with pytest.raises(SlurmJobError, match="Config file not found"):
            config_processor.load_and_merge_configs("nonexistent.yaml")

    def test_load_config_yaml_error(self, config_processor):
        """Test error handling for malformed YAML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp_file:
            tmp_file.write("invalid: yaml: content: [")
            tmp_file.flush()

            with pytest.raises(SlurmJobError, match="Invalid YAML"):
                config_processor.load_and_merge_configs(tmp_file.name)

        os.unlink(tmp_file.name)

    def test_load_config_include_file_not_found(self, config_processor):
        """Test error handling for missing include file."""
        main_data = {"training": {"__include": ["nonexistent.yaml"]}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as main_file:
            yaml.dump(main_data, main_file)
            main_file.flush()

            with pytest.raises(SlurmJobError, match="Include file not found"):
                config_processor.load_and_merge_configs(main_file.name)

        os.unlink(main_file.name)

    def test_load_config_include_yaml_error(self, config_processor):
        """Test error handling for malformed include YAML."""
        # Create malformed include file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as include_file:
            include_file.write("invalid: yaml: [")
            include_file.flush()

            main_data = {"training": {"__include": [include_file.name]}}
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as main_file:
                yaml.dump(main_data, main_file)
                main_file.flush()

                with pytest.raises(SlurmJobError, match="Invalid YAML"):
                    config_processor.load_and_merge_configs(main_file.name)

        os.unlink(include_file.name)
        os.unlink(main_file.name)

    def test_replace_placeholders_string(self, config_processor):
        """Test string placeholder replacement."""
        config = {"model": "<<model_name>>"}
        replacements = {"model_name": "gpt2"}
        result = config_processor.replace_placeholders(config, replacements)
        assert result["model"] == "gpt2"

    def test_replace_placeholders_typed(self, config_processor):
        """Test typed placeholder replacement."""
        config = {"epochs": "<<int: num_epochs>>", "rate": "<<float: lr>>"}
        replacements = {"num_epochs": "10", "lr": "0.001"}
        result = config_processor.replace_placeholders(config, replacements)
        assert result["epochs"] == 10
        assert result["rate"] == 0.001

    def test_replace_placeholders_boolean(self, config_processor):
        """Test boolean placeholder replacement."""
        config = {"debug": "<<bool: debug_mode>>"}
        replacements = {"debug_mode": "true"}
        result = config_processor.replace_placeholders(config, replacements)
        assert result["debug"] is True

    def test_replace_placeholders_list_int(self, config_processor):
        """Test list integer placeholder replacement."""
        config = {"layers": "<<list_int: layer_sizes>>"}
        replacements = {"layer_sizes": "128, 64, 32"}
        result = config_processor.replace_placeholders(config, replacements)
        assert result["layers"] == [128, 64, 32]

    def test_replace_placeholders_invalid_conversion(self, config_processor):
        """Test error handling for invalid type conversion."""
        config = {"value": "<<int: invalid_int>>"}
        replacements = {"invalid_int": "not_a_number"}
        result = config_processor.replace_placeholders(config, replacements)
        assert result["value"] is None

    def test_replace_placeholders_mixed_content(self, config_processor):
        """Test placeholder replacement in mixed content."""
        config = {"message": "Model <<model>> has <<int: layers>> layers"}
        replacements = {"model": "gpt2", "layers": "12"}
        result = config_processor.replace_placeholders(config, replacements)
        assert result["message"] == "Model gpt2 has 12 layers"

    def test_replace_placeholders_with_list(self, config_processor):
        """Test placeholder replacement in list structures."""
        config = {
            "models": ["<<model1>>", "<<model2>>"],
            "params": [
                {"lr": "<<float: learning_rate>>"},
                {"epochs": "<<int: num_epochs>>"},
            ],
        }
        replacements = {
            "model1": "gpt2",
            "model2": "bert",
            "learning_rate": "0.001",
            "num_epochs": "10",
        }
        result = config_processor.replace_placeholders(config, replacements)

        assert result["models"] == ["gpt2", "bert"]
        assert result["params"][0]["lr"] == 0.001
        assert result["params"][1]["epochs"] == 10

    def test_replace_placeholders_non_string_types(self, config_processor):
        """Test placeholder replacement with non-string types."""
        config = {
            "number": 42,
            "boolean": True,
            "none_value": None,
            "mixed": {"string": "<<value>>", "number": 123},
        }
        replacements = {"value": "replaced"}
        result = config_processor.replace_placeholders(config, replacements)

        # Non-string types should remain unchanged
        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["none_value"] is None
        assert result["mixed"]["string"] == "replaced"
        assert result["mixed"]["number"] == 123

    def test_replace_placeholders_no_replacement_found(self, config_processor):
        """Test placeholder behavior when no replacement is found."""
        config = {"value": "<<missing_placeholder>>"}
        replacements = {"other_key": "value"}
        result = config_processor.replace_placeholders(config, replacements)

        # Placeholder should remain unchanged when no replacement found
        assert result["value"] == "<<missing_placeholder>>"


class TestResourceCalculator:
    """Test suite for ResourceCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create ResourceCalculator instance for testing."""
        return ResourceCalculator()

    def test_calculate_resources_single_node(self, calculator):
        """Test resource calculation for single node."""
        result = calculator.calculate_resources(2)
        expected = {
            "n_nodes": 1,
            "n_gpu": 2,
            "n_cpu": 36,  # 2 * 18
            "partition": "gpu",
            "memory": 250000,  # 2 * 125000
        }
        assert result == expected

    def test_calculate_resources_full_node(self, calculator):
        """Test resource calculation for full node (4 GPUs)."""
        result = calculator.calculate_resources(4)
        expected = {
            "n_nodes": 1,
            "n_gpu": 4,
            "n_cpu": 72,  # 4 * 18
            "partition": "gpu",
            "memory": 0,  # Full node, no memory limit
        }
        assert result == expected

    def test_calculate_resources_multi_node(self, calculator):
        """Test resource calculation for multiple nodes."""
        result = calculator.calculate_resources(8)
        expected = {
            "n_nodes": 2,
            "n_gpu": 4,  # GPUs per node
            "n_cpu": 72,  # 4 * 18
            "partition": "gpu",
            "memory": 0,  # Full nodes
        }
        assert result == expected

    def test_calculate_resources_invalid_multi_node(self, calculator):
        """Test error for invalid multi-node GPU count."""
        with pytest.raises(
            SlurmJobError, match="GPU count must be divisible by 4"
        ):
            calculator.calculate_resources(6)

    def test_calculate_resources_single_gpu(self, calculator):
        """Test resource calculation for single GPU."""
        result = calculator.calculate_resources(1)
        expected = {
            "n_nodes": 1,
            "n_gpu": 1,
            "n_cpu": 18,  # 1 * 18
            "partition": "gpu",
            "memory": 125000,  # 1 * 125000
        }
        assert result == expected


class TestFileManager:
    """Test suite for FileManager class."""

    @pytest.fixture
    def file_manager(self):
        """Create FileManager instance for testing."""
        return FileManager()

    def test_ensure_output_directory_default(self, file_manager):
        """Test creating default output directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            result = file_manager.ensure_output_directory(
                None, "test_group", "test_job", "2024_01_01__12_00_00"
            )
            expected_path = Path(
                "experiments/test_group/test_job/2024_01_01__12_00_00"
            )
            assert result == expected_path
            assert result.exists()

    def test_ensure_output_directory_custom(self, file_manager):
        """Test creating custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_path = Path(temp_dir) / "custom_output"
            result = file_manager.ensure_output_directory(
                str(custom_path), "group", "job", "id"
            )
            assert result == custom_path
            assert result.exists()

    def test_copy_config_to_output(self, file_manager):
        """Test copying configuration to output directory."""
        config = {"model": "gpt2", "epochs": 10}
        job_id = "test_job_id"

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result_path = file_manager.copy_config_to_output(
                config, output_dir, job_id
            )

            expected_path = output_dir / f"{job_id}.yml"
            assert result_path == expected_path
            assert result_path.exists()

            # Verify content
            with open(result_path) as f:
                loaded_config = yaml.load(f, Loader=yaml.FullLoader)
            assert loaded_config == config

    def test_generate_slurm_script(self, file_manager):
        """Test SLURM script generation from template."""
        template_content = (
            "#!/bin/bash\n#SBATCH --job-name={job_name}\necho {message}"
        )
        script_args = {"job_name": "test_job", "message": "Hello World"}

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create template file
            template_path = Path(temp_dir) / "template.slurm"
            with open(template_path, "w") as f:
                f.write(template_content)

            output_dir = Path(temp_dir)
            job_id = "test_job_id"

            result_path = file_manager.generate_slurm_script(
                template_path, output_dir, job_id, script_args
            )

            expected_path = output_dir / f"{job_id}.sh"
            assert result_path == expected_path
            assert result_path.exists()

            # Verify content
            with open(result_path) as f:
                content = f.read()
            expected_content = (
                "#!/bin/bash\n#SBATCH --job-name=test_job\necho Hello World"
            )
            assert content == expected_content

    def test_generate_slurm_script_missing_template(self, file_manager):
        """Test error handling for missing template file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with pytest.raises(SlurmJobError, match="Template file not found"):
                file_manager.generate_slurm_script(
                    "nonexistent.slurm", output_dir, "job_id", {}
                )

    def test_generate_slurm_script_missing_variable(self, file_manager):
        """Test error handling for missing template variable."""
        template_content = "#!/bin/bash\n#SBATCH --job-name={missing_var}"

        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "template.slurm"
            with open(template_path, "w") as f:
                f.write(template_content)

            output_dir = Path(temp_dir)
            script_args = {"job_name": "test"}  # missing_var not provided

            with pytest.raises(
                SlurmJobError, match="Missing template variable"
            ):
                file_manager.generate_slurm_script(
                    template_path, output_dir, "job_id", script_args
                )


class TestSlurmSubmitter:
    """Test suite for SlurmSubmitter class."""

    @pytest.fixture
    def submitter(self):
        """Create SlurmSubmitter instance for testing."""
        return SlurmSubmitter()

    def test_submit_job_dry_run(self, submitter, capsys):
        """Test dry run mode (no actual submission)."""
        script_path = Path("/tmp/test_script.sh")
        submitter.submit_job(script_path, dry_run=True)

        captured = capsys.readouterr()
        assert "Generated script at" in captured.out
        assert "without submission" in captured.out

    @patch("subprocess.run")
    def test_submit_job_success(self, mock_run, submitter, capsys):
        """Test successful job submission."""
        mock_run.return_value = MagicMock(
            stdout="Submitted batch job 12345\n", returncode=0
        )

        script_path = Path("/tmp/test_script.sh")
        submitter.submit_job(script_path, dry_run=False)

        mock_run.assert_called_once_with(
            ["sbatch", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        captured = capsys.readouterr()
        assert "Job submitted successfully" in captured.out

    @patch("subprocess.run")
    def test_submit_job_failure(self, mock_run, submitter):
        """Test job submission failure."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "sbatch", stderr="Permission denied"
        )

        script_path = Path("/tmp/test_script.sh")
        with pytest.raises(SlurmJobError, match="Failed to submit job"):
            submitter.submit_job(script_path, dry_run=False)

    @patch("subprocess.run")
    def test_submit_job_command_not_found(self, mock_run, submitter):
        """Test error when sbatch command is not found."""
        mock_run.side_effect = FileNotFoundError("sbatch: command not found")

        script_path = Path("/tmp/test_script.sh")
        with pytest.raises(SlurmJobError, match="sbatch command not found"):
            submitter.submit_job(script_path, dry_run=False)


class TestSlurmJobManager:
    """Test suite for SlurmJobManager class."""

    @pytest.fixture
    def job_manager(self):
        """Create SlurmJobManager instance for testing."""
        return SlurmJobManager()

    def test_create_argument_parser(self, job_manager):
        """Test argument parser creation."""
        parser = job_manager.create_argument_parser()
        assert isinstance(parser, argparse.ArgumentParser)

        # Test parsing required argument
        args = parser.parse_args(["--config_file", "test.yaml"])
        assert args.config_file == "test.yaml"

    def test_process_arguments_basic(self, job_manager):
        """Test basic argument processing."""
        argv = ["script.py", "--config_file", "test.yaml", "--n_gpu", "2"]
        args_dict, dynamic_args = job_manager.process_arguments(argv)

        assert args_dict["config_file"] == "test.yaml"
        assert args_dict["n_gpu"] == 2
        assert dynamic_args == {}

    def test_process_arguments_with_dynamic(self, job_manager):
        """Test argument processing with dynamic arguments."""
        argv = [
            "script.py",
            "--config_file",
            "test.yaml",
            "--custom_param",
            "value123",
        ]
        args_dict, dynamic_args = job_manager.process_arguments(argv)

        assert args_dict["config_file"] == "test.yaml"
        assert args_dict["custom_param"] == "value123"
        assert dynamic_args == {"custom_param": "value123"}

    @patch("run_slurm.SlurmJobManager.process_arguments")
    @patch("run_slurm.ConfigProcessor.load_and_merge_configs")
    @patch("run_slurm.FileManager.ensure_output_directory")
    def test_run_method_error_handling(
        self, mock_ensure_dir, mock_load_config, mock_process_args, job_manager
    ):
        """Test error handling in run method."""
        mock_process_args.side_effect = SlurmJobError("Test error")

        with pytest.raises(SystemExit):
            job_manager.run(["script.py", "--config_file", "test.yaml"])

    @patch("run_slurm.SlurmSubmitter.submit_job")
    @patch("run_slurm.FileManager.generate_slurm_script")
    @patch("run_slurm.FileManager.copy_config_to_output")
    @patch("run_slurm.FileManager.ensure_output_directory")
    @patch("run_slurm.ConfigProcessor.replace_placeholders")
    @patch("run_slurm.ConfigProcessor.load_and_merge_configs")
    @patch("run_slurm.ResourceCalculator.calculate_resources")
    @patch("run_slurm.ConfigProcessor.generate_job_id")
    def test_run_method_complete_workflow(
        self,
        mock_job_id,
        mock_calc_resources,
        mock_load_config,
        mock_replace_placeholders,
        mock_ensure_dir,
        mock_copy_config,
        mock_generate_script,
        mock_submit,
        job_manager,
    ):
        """Test complete workflow of run method."""
        # Setup mocks
        mock_job_id.return_value = "2024_01_01__12_00_00"
        mock_calc_resources.return_value = {
            "n_nodes": 1,
            "n_gpu": 1,
            "n_cpu": 18,
            "partition": "gpu",
            "memory": 125000,
        }
        mock_load_config.return_value = {"model": "gpt2"}
        mock_replace_placeholders.return_value = {"model": "gpt2"}
        mock_ensure_dir.return_value = Path("/tmp/output")
        mock_copy_config.return_value = Path("/tmp/output/config.yml")
        mock_generate_script.return_value = Path("/tmp/output/script.sh")

        argv = ["script.py", "--config_file", "test.yaml"]

        # Should not raise any exceptions
        job_manager.run(argv)

        # Verify all steps were called
        mock_job_id.assert_called_once()
        mock_calc_resources.assert_called_once()
        mock_load_config.assert_called_once()
        mock_replace_placeholders.assert_called_once()
        mock_ensure_dir.assert_called_once()
        mock_copy_config.assert_called_once()
        mock_generate_script.assert_called_once()
        mock_submit.assert_called_once()

    def test_run_method_unexpected_error(self, job_manager):
        """Test handling of unexpected errors in run method."""
        # Force an unexpected error
        with patch.object(
            job_manager,
            "process_arguments",
            side_effect=RuntimeError("Unexpected error"),
        ):
            with pytest.raises(SystemExit):
                job_manager.run(["script.py", "--config_file", "test.yaml"])
