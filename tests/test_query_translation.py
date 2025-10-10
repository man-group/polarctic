from polarctic.polarctic import PolarsToArcticDBTranslator
from arcticdb import QueryBuilder
import polars as pl
import pytest

def test_col(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") > 2, q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] > 2]

    assert q == qe
