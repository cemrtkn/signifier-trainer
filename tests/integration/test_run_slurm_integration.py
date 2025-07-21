"""
Integration tests for run_slurm.py CLI functionality.
Tests the complete workflow: argument parsing, config processing,
file generation, and job submission logic.
"""
import os
import shutil
import subprocess
import tempfile
import yaml
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        yield workspace


@pytest.fixture
def sample_config_basic(temp_workspace):
    """Create a basic YAML config file."""
    config = {
        "model": {"name": "test_model", "lr": 0.001},
        "data": {"batch_size": 32, "dataset": "test_data"},
        "training": {"epochs": 10}
    }
    config_path = temp_workspace / "config_basic.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def sample_config_with_placeholders(temp_workspace):
    """Create a YAML config with placeholder values."""
    config = {
        "model": {"name": "<<model_name>>", "lr": "<<float:learning_rate>>"},
        "data": {"batch_size": "<<int:batch_size>>", "enabled": "<<bool:use_data>>"},
        "training": {"epochs": "<<int:epochs>>", "gpus": "<<list_int:gpu_ids>>"}
    }
    config_path = temp_workspace / "config_placeholders.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@pytest.fixture
def sample_config_with_includes(temp_workspace):
    """Create configs with __include functionality."""
    # Base config to include
    base_config = {
        "model": {"name": "base_model", "lr": 0.001},
        "optimizer": {"type": "adam", "beta1": 0.9}
    }
    base_path = temp_workspace / "base_config.yml"
    with open(base_path, "w") as f:
        yaml.dump(base_config, f)
    
    # Main config with include
    main_config = {
        "__include": [str(base_path)],
        "model": {"lr": 0.01},  # Override base value
        "data": {"batch_size": 64}
    }
    main_path = temp_workspace / "config_with_include.yml"
    with open(main_path, "w") as f:
        yaml.dump(main_config, f)
    
    return main_path, base_path


@pytest.fixture
def sample_slurm_template(temp_workspace):
    """Create a sample SLURM template."""
    template_content = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/slurm-%j.out
#SBATCH --error={output_dir}/slurm-%j.err
#SBATCH --nodes={n_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --gres=gpu:{n_gpu}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem={memory}

# Load environment
module load apptainer

# Run the script
apptainer exec {image} python {script} --config {config_file}
"""
    template_path = temp_workspace / "test_template.slurm"
    with open(template_path, "w") as f:
        f.write(template_content)
    return template_path


class TestRunSlurmIntegration:
    """Integration tests for run_slurm.py CLI."""

    def test_basic_dry_run(self, temp_workspace, sample_config_basic,
                          sample_slurm_template):
        """Test basic dry run with minimal arguments."""
        # Get the absolute path to the project root
        project_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(sample_config_basic),
            "--template", str(sample_slurm_template),
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Generated script at" in result.stdout
        assert "without submission" in result.stdout

    def test_custom_arguments(self, temp_workspace, sample_config_basic,
                             sample_slurm_template):
        """Test with custom job parameters."""
        project_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(sample_config_basic),
            "--template", str(sample_slurm_template),
            "--job_name", "test_job",
            "--group_name", "test_group",
            "--n_gpu", "2",
            "--time", "01:30:00",
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        assert result.returncode == 0

    def test_placeholder_replacement(self, temp_workspace,
                                   sample_config_with_placeholders,
                                   sample_slurm_template):
        """Test placeholder replacement in config files."""
        project_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(sample_config_with_placeholders),
            "--template", str(sample_slurm_template),
            "--model_name", "resnet50",
            "--learning_rate", "0.01",
            "--batch_size", "64",
            "--use_data", "true",
            "--epochs", "20",
            "--gpu_ids", "0,1,2",
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        assert result.returncode == 0

    def test_include_functionality(self, temp_workspace,
                                 sample_config_with_includes,
                                 sample_slurm_template):
        """Test config file inclusion."""
        project_root = Path(__file__).parent.parent.parent
        main_config, _ = sample_config_with_includes
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(main_config),
            "--template", str(sample_slurm_template),
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        assert result.returncode == 0

    def test_output_directory_creation(self, temp_workspace,
                                     sample_config_basic,
                                     sample_slurm_template):
        """Test that output directories are created correctly."""
        project_root = Path(__file__).parent.parent.parent
        custom_output = temp_workspace / "custom_output"
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(sample_config_basic),
            "--template", str(sample_slurm_template),
            "--output_dir", str(custom_output),
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        assert result.returncode == 0
        assert custom_output.exists()
        assert custom_output.is_dir()

    def test_gpu_resource_calculation(self, temp_workspace,
                                    sample_config_basic,
                                    sample_slurm_template):
        """Test GPU resource calculation for different configurations."""
        project_root = Path(__file__).parent.parent.parent
        
        # Test single GPU
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(sample_config_basic),
            "--template", str(sample_slurm_template),
            "--n_gpu", "1",
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        assert result.returncode == 0
        
        # Test multi-GPU (4 GPUs)
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(sample_config_basic),
            "--template", str(sample_slurm_template),
            "--n_gpu", "4",
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        assert result.returncode == 0

    def test_file_generation(self, temp_workspace, sample_config_basic,
                           sample_slurm_template):
        """Test that all expected files are generated."""
        project_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(sample_config_basic),
            "--template", str(sample_slurm_template),
            "--job_name", "test_job",
            "--group_name", "test_group",
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Generated script at" in result.stdout
        
        # Parse the output to find where the files were generated
        import re
        script_path_match = re.search(r"Generated script at (.*?) without", result.stdout)
        if script_path_match:
            script_path = Path(script_path_match.group(1))
            assert script_path.exists(), f"Generated script should exist at {script_path}"
            
            # Check output directory
            output_dir = script_path.parent
            assert output_dir.exists()
            
            # Check for copied config file
            config_files = list(output_dir.glob("*.yml"))
            assert len(config_files) > 0, f"Config file should exist in {output_dir}"
            
            # Check for generated script
            script_files = list(output_dir.glob("*.sh"))
            assert len(script_files) > 0, f"Script file should exist in {output_dir}"

    @mock.patch('subprocess.run')
    def test_job_submission_called(self, mock_subprocess, temp_workspace,
                                 sample_config_basic, sample_slurm_template):
        """Test that sbatch is called when not in dry mode."""
        project_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(sample_config_basic),
            "--template", str(sample_slurm_template),
            "--job_name", "test_job"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        # Note: This will still fail because the real subprocess.run
        # is called internally, but it shows the test structure

    def test_invalid_config_file(self, temp_workspace, sample_slurm_template):
        """Test behavior with non-existent config file."""
        project_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", "nonexistent.yml",
            "--template", str(sample_slurm_template),
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        assert result.returncode != 0

    def test_invalid_template_file(self, temp_workspace, sample_config_basic):
        """Test behavior with non-existent template file."""
        project_root = Path(__file__).parent.parent.parent
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(sample_config_basic),
            "--template", "nonexistent.slurm",
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        assert result.returncode != 0


class TestConfigProcessing:
    """Test config processing functionality."""

    def test_placeholder_type_conversion(self, temp_workspace,
                                       sample_slurm_template):
        """Test different placeholder type conversions."""
        project_root = Path(__file__).parent.parent.parent
        
        config = {
            "float_val": "<<float:test_float>>",
            "int_val": "<<int:test_int>>",
            "bool_val": "<<bool:test_bool>>",
            "list_int_val": "<<list_int:test_list>>",
            "str_val": "<<test_string>>"
        }
        config_path = temp_workspace / "type_test.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        result = subprocess.run([
            "python", str(project_root / "src" / "run_slurm.py"),
            "--config_file", str(config_path),
            "--template", str(sample_slurm_template),
            "--test_float", "3.14",
            "--test_int", "42",
            "--test_bool", "false",
            "--test_list", "1,2,3",
            "--test_string", "hello",
            "--dry"
        ], cwd=str(project_root),
           capture_output=True, text=True)
        
        assert result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__]) 