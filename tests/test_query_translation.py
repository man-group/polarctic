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

    q2 = QueryBuilder()
    q2 = translator.translate((pl.col("col1") > 2).and_(pl.col("col2") < 3), q2)
    assert q2 == qe


def test_or(translator):
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") > 2) | (pl.col("col2") < 3), q)
    qe = QueryBuilder()
    qe = qe[(qe["col1"] > 2) | (qe["col2"] < 3)]
    assert q == qe

    q2 = QueryBuilder()
    q2 = translator.translate((pl.col("col1") > 2).or_(pl.col("col2") < 3), q2)
    assert q2 == qe

def test_not(translator):
    q = QueryBuilder()
    q = translator.translate(~pl.col("col1"), q)
    qe = QueryBuilder()
    qe = qe[~qe["col1"]]
    assert q == qe

def test_abs(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1").abs(), q)
    qe = QueryBuilder()
    qe = qe[abs(qe["col1"])]
    assert q == qe

def test_is_null(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1").is_null(), q)
    qe = QueryBuilder()
    qe = qe[qe["col1"].isnull()]
    assert q == qe

def test_is_not_null(translator):
    q = QueryBuilder()
    q = translator.translate(~pl.col("col1").is_null(), q)
    qe = QueryBuilder()
    qe = qe[~qe["col1"].isnull()]
    assert q == qe
    qe2 = QueryBuilder()
    qe2 = qe2[qe2["col1"].notnull()]
    assert(qe2 != qe)

def test_regex_match(translator):
    q = QueryBuilder()
    q = translator.translate(pl.col("col1").str.contains("e+"), q)
    qe = QueryBuilder()
    qe = qe[qe["col1"].regex_match("e+")]
    assert q == qe

