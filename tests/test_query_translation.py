from polarctic.polarctic import PolarsToArcticDBTranslator
from arcticdb import QueryBuilder
import polars as pl
import pytest

def test_greater(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") > 2, q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] > 2]
    assert q == qe

def test_greater_equal(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") >= 2, q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] >= 2]
    assert q == qe

def test_less(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") < 2, q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] < 2]
    assert q == qe

def test_less_equal(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") <= 2, q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] <= 2]
    assert q == qe

def test_equal(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") == 2, q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] == 2]
    assert q == qe

def test_not_equal(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") != 2, q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] != 2]
    assert q == qe

def test_add(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") + pl.col("col2"), q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] + qe["col2"]]
    assert q == qe

def test_subtract(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") - pl.col("col2"), q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] - qe["col2"]]
    assert q == qe

def test_multiply(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") * pl.col("col2"), q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] * qe["col2"]]
    assert q == qe

def test_divide(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1") / pl.col("col2"), q)
    qe = QueryBuilder()
    qe = qe[qe["col1"] / qe["col2"]]
    assert q == qe

def test_modulo(translator):
    q = QueryBuilder()
    with pytest.raises(NotImplementedError):
        q = translator.translate(pl.col("col1") % pl.col("col2"), q)

def test_negation(translator):
    q = QueryBuilder()
    q = translator.translate(-pl.col("col1"), q)
    qe = QueryBuilder()
    qe = qe[-qe["col1"]]
    assert q == qe

def test_and(translator):
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") > 2) & (pl.col("col2") < 3), q)
    qe = QueryBuilder()
    qe = qe[(qe["col1"] > 2) & (qe["col2"] < 3)]
    assert q == qe


def test_or(translator):
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") > 2) | (pl.col("col2") < 3), q)
    qe = QueryBuilder()
    qe = qe[(qe["col1"] > 2) | (qe["col2"] < 3)]
    assert q == qe

# TODO: fix
#def test_not(translator):
#    q = QueryBuilder()
#    q = translator.translate(~pl.col("col1"), q)
#    qe = QueryBuilder()
#    qe = qe[~qe["col1"]]
#    assert q == qe

