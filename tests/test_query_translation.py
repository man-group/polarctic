from typing import Any

import polars as pl
import pytest
from arcticdb import QueryBuilder

from polarctic.polarctic import PolarsToArcticDBTranslator

"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""


def make_query_builder() -> Any:
    return QueryBuilder()


def test_greater(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") > 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] > 2]
    assert q == qe


def test_greater_equal(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") >= 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] >= 2]
    assert q == qe


def test_less(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") < 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] < 2]
    assert q == qe


def test_less_equal(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") <= 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] <= 2]
    assert q == qe


def test_equal(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") == 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] == 2]
    assert q == qe


def test_not_equal(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") != 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] != 2]
    assert q == qe


def test_add(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") + pl.col("col2"), q)
    qe = make_query_builder()
    qe = qe[qe["col1"] + qe["col2"]]
    assert q == qe


def test_subtract(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") - pl.col("col2"), q)
    qe = make_query_builder()
    qe = qe[qe["col1"] - qe["col2"]]
    assert q == qe


def test_multiply(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") * pl.col("col2"), q)
    qe = make_query_builder()
    qe = qe[qe["col1"] * qe["col2"]]
    assert q == qe


def test_divide(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1") / pl.col("col2"), q)
    qe = make_query_builder()
    qe = qe[qe["col1"] / qe["col2"]]
    assert q == qe


def test_modulo(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    with pytest.raises(NotImplementedError):
        q = translator.translate(pl.col("col1") % pl.col("col2"), q)


def test_negation(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(-pl.col("col1"), q)
    qe = make_query_builder()
    qe = qe[-qe["col1"]]
    assert q == qe


def test_and(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate((pl.col("col1") > 2) & (pl.col("col2") < 3), q)
    qe = make_query_builder()
    qe = qe[(qe["col1"] > 2) & (qe["col2"] < 3)]
    assert q == qe

    q2 = make_query_builder()
    q2 = translator.translate((pl.col("col1") > 2).and_(pl.col("col2") < 3), q2)
    assert q2 == qe


def test_bitwise_and_integer(translator: PolarsToArcticDBTranslator) -> None:
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") & 1) == 0, q)
    qe: Any = QueryBuilder()
    qe = qe[(qe["col1"] & 1) == 0]
    assert q == qe


def test_or(translator):
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") > 2) | (pl.col("col2") < 3), q)
    qe = make_query_builder()
    qe = qe[(qe["col1"] > 2) | (qe["col2"] < 3)]
    assert q == qe

    q2 = make_query_builder()
    q2 = translator.translate((pl.col("col1") > 2).or_(pl.col("col2") < 3), q2)
    assert q2 == qe


def test_bitwise_or_integer(translator: PolarsToArcticDBTranslator) -> None:
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") | 1) == 3, q)
    qe: Any = QueryBuilder()
    qe = qe[(qe["col1"] | 1) == 3]
    assert q == qe


def test_bitwise_xor_integer(translator: PolarsToArcticDBTranslator) -> None:
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") ^ 1) == 3, q)
    qe: Any = QueryBuilder()
    qe = qe[(qe["col1"] ^ 1) == 3]
    assert q == qe


def test_not(translator):
    q = QueryBuilder()
    q = translator.translate(~pl.col("col1"), q)
    qe = make_query_builder()
    qe = qe[~qe["col1"]]
    assert q == qe


def test_abs(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1").abs(), q)
    qe = make_query_builder()
    qe = qe[abs(qe["col1"])]
    assert q == qe


def test_is_null(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1").is_null(), q)
    qe = make_query_builder()
    qe = qe[qe["col1"].isnull()]
    assert q == qe


def test_is_not_null(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(~pl.col("col1").is_null(), q)
    qe = make_query_builder()
    qe = qe[~qe["col1"].isnull()]
    assert q == qe
    qe2 = make_query_builder()
    qe2 = qe2[qe2["col1"].notnull()]
    assert qe2 != qe


def test_regex_match(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1").str.contains("e+"), q)
    qe = make_query_builder()
    qe = qe[qe["col1"].regex_match("e+")]
    assert q == qe


def test_isin(translator: PolarsToArcticDBTranslator) -> None:
    q = make_query_builder()
    q = translator.translate(pl.col("col1").is_in([24, 42]), q)
    qe = make_query_builder()
    qe = qe[qe["col1"].isin(24, 42)]
    assert q == qe
