import pytest

from example_func import ExampleClass


@pytest.fixture
def example_instance():
    """Fixture for ExampleClass with custom name."""
    return ExampleClass("test_instance")


@pytest.fixture
def default_instance():
    """Fixture for ExampleClass with default name."""
    return ExampleClass()


def test_init_with_name(example_instance):
    """Test ExampleClass initialization with custom name."""
    assert example_instance.name == "test_instance"


def test_init_default_name(default_instance):
    """Test ExampleClass initialization with default name."""
    assert default_instance.name == "default"


def test_example_method_with_custom_name(example_instance):
    """Test example_method returns correct format with custom name."""
    expected = "example_method executed by test_instance"
    assert example_instance.example_method() == expected


def test_example_method_with_default_name(default_instance):
    """Test example_method returns correct format with default name."""
    expected = "example_method executed by default"
    assert default_instance.example_method() == expected


def test_get_name(example_instance, default_instance):
    """Test get_name returns the correct name."""
    assert example_instance.get_name() == "test_instance"
    assert default_instance.get_name() == "default"
