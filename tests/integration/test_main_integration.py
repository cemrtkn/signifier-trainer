from example_func import ExampleClass
from main import main


def test_main_function_output(capsys):
    """Test that main function produces expected output."""
    result = main()
    captured = capsys.readouterr()

    # Check that the function returns the expected result
    expected_result = "example_method executed by MainApp"
    assert result == expected_result

    # Check that the output contains expected strings
    assert "Result: example_method executed by MainApp" in captured.out
    assert "Instance name: MainApp" in captured.out


def test_main_function_creates_correct_instance():
    """Test that main function creates ExampleClass with correct name."""
    # This test verifies the integration between main and ExampleClass
    result = main()
    assert result == "example_method executed by MainApp"


def test_full_workflow_integration():
    """Test the complete workflow integration."""
    # Create instance like main() does
    instance = ExampleClass("MainApp")

    # Execute method like main() does
    result = instance.example_method()
    name = instance.get_name()

    # Verify the integration works as expected
    assert result == "example_method executed by MainApp"
    assert name == "MainApp"
