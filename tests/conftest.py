import pytest
from polarctic.polarctic import PolarsToArcticDBTranslator

@pytest.fixture
def translator():
    return PolarsToArcticDBTranslator()
