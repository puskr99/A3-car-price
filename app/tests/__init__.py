# tests/__init__.py
import pytest

@pytest.fixture
def global_fixture():
    return "Shared resource"