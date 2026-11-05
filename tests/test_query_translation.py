"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Tests 3: check translation of Polars filter expressions into ArcticDB queries and handling of unsupported expressions."""

import ast
from typing import Any

import polars as pl
import pytest
from arcticdb import QueryBuilder

from polarctic.polarctic import PolarsToArcticDBTranslator

from arcticdb.version_store.processing import ExpressionNode
from arcticdb_ext.version_store import OperationType


def make_query_builder() -> Any:
    return QueryBuilder()


def test_greater(translator: PolarsToArcticDBTranslator) -> None:
    """Check construction of the ArcticDB equivalent of a Polars greater-than condition."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1") > 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] > 2]
    assert q == qe


def test_greater_equal(translator: PolarsToArcticDBTranslator) -> None:
    """Check construction of the ArcticDB equivalent of a Polars greater-than-or-equal condition."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1") >= 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] >= 2]
    assert q == qe


def test_less(translator: PolarsToArcticDBTranslator) -> None:
    """Check construction of the ArcticDB equivalent of a Polars less-than condition."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1") < 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] < 2]
    assert q == qe


def test_less_equal(translator: PolarsToArcticDBTranslator) -> None:
    """Check construction of the ArcticDB equivalent of a Polars less-than-or-equal condition."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1") <= 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] <= 2]
    assert q == qe


def test_equal(translator: PolarsToArcticDBTranslator) -> None:
    """Check construction of the ArcticDB equivalent of a Polars equality condition."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1") == 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] == 2]
    assert q == qe


def test_not_equal(translator: PolarsToArcticDBTranslator) -> None:
    """Check construction of the ArcticDB equivalent of a Polars inequality condition."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1") != 2, q)
    qe = make_query_builder()
    qe = qe[qe["col1"] != 2]
    assert q == qe


def test_add(translator: PolarsToArcticDBTranslator) -> None:
    """Check translation of addition inside a Boolean filter, not creation of an output column."""
    q = make_query_builder()
    q = translator.translate((pl.col("col1") + pl.col("col2")) > 20, q)
    qe = make_query_builder()
    qe = qe[(qe["col1"] + qe["col2"]) > 20]
    assert q == qe


def test_subtract(translator: PolarsToArcticDBTranslator) -> None:
    """Check translation of subtraction inside a Boolean filter."""
    q = make_query_builder()
    q = translator.translate((pl.col("col1") - pl.col("col2")) > 0, q)
    qe = make_query_builder()
    qe = qe[(qe["col1"] - qe["col2"]) > 0]
    assert q == qe


def test_multiply(translator: PolarsToArcticDBTranslator) -> None:
    """Check translation of multiplication inside a Boolean filter."""
    q = make_query_builder()
    q = translator.translate((pl.col("col1") * pl.col("col2")) > 20, q)
    qe = make_query_builder()
    qe = qe[(qe["col1"] * qe["col2"]) > 20]
    assert q == qe


def test_divide(translator: PolarsToArcticDBTranslator) -> None:
    """Check translation of division inside a Boolean filter."""
    q = make_query_builder()
    q = translator.translate((pl.col("col1") / pl.col("col2")) > 2, q)
    qe = make_query_builder()
    qe = qe[(qe["col1"] / qe["col2"]) > 2]
    assert q == qe


def test_modulo(translator: PolarsToArcticDBTranslator) -> None:
    """Check that the current translator explicitly rejects remainder arithmetic."""
    q = make_query_builder()
    with pytest.raises(NotImplementedError):
        q = translator.translate(pl.col("col1") % pl.col("col2"), q)


def test_negation(translator: PolarsToArcticDBTranslator) -> None:
    """Check translation of numeric negation inside a Boolean filter."""
    q = make_query_builder()
    q = translator.translate(-pl.col("col1") < 0, q)
    qe = make_query_builder()
    qe = qe[-qe["col1"] < 0]
    assert q == qe


def test_and(translator: PolarsToArcticDBTranslator) -> None:
    """Check that operator and method spellings of Boolean AND construct the same ArcticDB filter."""
    q = make_query_builder()
    q = translator.translate((pl.col("col1") > 2) & (pl.col("col2") < 3), q)
    qe = make_query_builder()
    qe = qe[(qe["col1"] > 2) & (qe["col2"] < 3)]
    assert q == qe

    q2 = make_query_builder()
    q2 = translator.translate((pl.col("col1") > 2).and_(pl.col("col2") < 3), q2)
    assert q2 == qe


def test_bitwise_and_integer(translator: PolarsToArcticDBTranslator) -> None:
    """Check integer bitwise-AND query construction; execution support is checked separately."""
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") & 1) == 0, q)
    qe: Any = QueryBuilder()
    qe = qe[(qe["col1"] & 1) == 0]
    assert q == qe


def test_or(translator: PolarsToArcticDBTranslator) -> None:
    """Check that operator and method spellings of Boolean OR construct the same ArcticDB filter."""
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") > 2) | (pl.col("col2") < 3), q)
    qe = make_query_builder()
    qe = qe[(qe["col1"] > 2) | (qe["col2"] < 3)]
    assert q == qe

    q2 = make_query_builder()
    q2 = translator.translate((pl.col("col1") > 2).or_(pl.col("col2") < 3), q2)
    assert q2 == qe


def test_bitwise_or_integer(translator: PolarsToArcticDBTranslator) -> None:
    """Check integer bitwise-OR query construction; execution support is checked separately."""
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") | 1) == 3, q)
    qe: Any = QueryBuilder()
    qe = qe[(qe["col1"] | 1) == 3]
    assert q == qe


def test_bitwise_xor_integer(translator: PolarsToArcticDBTranslator) -> None:
    """Check integer bitwise-XOR query construction; execution support is checked separately."""
    q = QueryBuilder()
    q = translator.translate((pl.col("col1") ^ 1) == 3, q)
    qe: Any = QueryBuilder()
    qe = qe[(qe["col1"] ^ 1) == 3]
    assert q == qe


def test_not(translator: PolarsToArcticDBTranslator) -> None:
    """Check construction of the ArcticDB equivalent of Polars Boolean negation."""
    q = QueryBuilder()
    q = translator.translate(~pl.col("col1"), q)
    qe = make_query_builder()
    qe = qe[~qe["col1"]]
    assert q == qe


def test_abs(translator: PolarsToArcticDBTranslator) -> None:
    """Check translation of absolute value inside a Boolean filter."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1").abs() > 2, q)
    qe = make_query_builder()
    qe = qe[abs(qe["col1"]) > 2]
    assert q == qe


def test_is_null(translator: PolarsToArcticDBTranslator) -> None:
    """Check that a Polars null check constructs the equivalent ArcticDB null-check expression."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1").is_null(), q)
    qe = make_query_builder()
    qe = qe[qe["col1"].isnull()]
    assert q == qe


def test_is_not_null(translator: PolarsToArcticDBTranslator) -> None:
    """Check construction of a negated null check; actual missing-value behaviour is tested separately."""
    q = make_query_builder()
    q = translator.translate(~pl.col("col1").is_null(), q)
    qe = make_query_builder()
    qe = qe[~qe["col1"].isnull()]
    assert q == qe


def test_regex_match(translator: PolarsToArcticDBTranslator) -> None:
    """Check that a Polars string regex condition maps to ArcticDB regex matching."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1").str.contains("e+"), q)
    qe = make_query_builder()
    qe = qe[qe["col1"].regex_match("e+")]
    assert q == qe


def test_isin(translator: PolarsToArcticDBTranslator) -> None:
    """Check that a Polars membership condition constructs the expected ArcticDB membership query."""
    q = make_query_builder()
    q = translator.translate(pl.col("col1").is_in([24, 42]), q)
    qe = make_query_builder()
    qe = qe[qe["col1"].isin(24, 42)]
    assert q == qe


def test_translate_invalid_expression_raises_value_error(
    translator: PolarsToArcticDBTranslator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check that a syntax-parsing failure becomes a ValueError with expression context."""
    # Force the parser error so this check is independent of Polars display formatting
    def raise_syntax_error(_expr: str) -> ast.AST:
        raise SyntaxError("invalid")

    monkeypatch.setattr(translator, "_parse_expression", raise_syntax_error)

    with pytest.raises(ValueError, match="Invalid Polars expression"):
        translator.translate(pl.col("col1"), QueryBuilder())


def test_replace_square_brackets_handles_missing_open_marker(
    translator: PolarsToArcticDBTranslator,
) -> None:
    """Check that bracket cleanup leaves text unchanged when its expected opening marker is absent."""
    assert translator._replace_square_brackets("miaow])") == "miaow])"


def test_process_node_returns_name(
    translator: PolarsToArcticDBTranslator,
) -> None:
    """Check that a name is returned unchanged."""
    node = ast.parse("field_name", mode="eval").body

    assert translator._process_node(node) == "field_name"


def test_process_node_rejects_lists(
    translator: PolarsToArcticDBTranslator,
) -> None:
    """Check that unsupported list expressions raise a clear error."""
    node = ast.parse("[1, 2]", mode="eval").body

    with pytest.raises(NotImplementedError, match="Node type List not supported"):
        translator._process_node(node)


def test_process_node_builds_numeric_negation(
    translator: PolarsToArcticDBTranslator,
) -> None:
    """Check that -1 is translated as negation of the number 1."""
    node = ast.parse("-1", mode="eval").body
    result = translator._process_node(node)

    assert isinstance(result, ExpressionNode)
    assert result.operator == OperationType.NEG
    assert result.left == 1
    assert result.right is None


def test_process_call_unsupported_methods_and_function_shapes(
    translator: PolarsToArcticDBTranslator,
) -> None:
    """Check that an unknown column method is explicitly rejected."""
    unsupported_method_node = ast.parse('col("a").miaow()', mode="eval").body
    assert isinstance(unsupported_method_node, ast.Call)
    with pytest.raises(NotImplementedError, match="Method miaow not supported"):
        translator._process_call(unsupported_method_node)


@pytest.mark.parametrize("expression", ['miaow("a")', "(lambda: 1)()"])
def test_process_call_rejects_unknown_function_shapes(translator, expression):
    """Check that unsupported function calls raise an explicit error instead of returning None."""
    node = ast.parse(expression, mode="eval").body
    with pytest.raises(NotImplementedError):
        translator._process_call(node)