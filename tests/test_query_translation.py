import ast
from typing import Any

import polars as pl
import pytest
from _pytest.monkeypatch import MonkeyPatch
from arcticdb import QueryBuilder

import polarctic.polarctic as polarctic_module
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


def test_or(translator: PolarsToArcticDBTranslator) -> None:
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


def test_not(translator: PolarsToArcticDBTranslator) -> None:
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


def test_translate_invalid_expression_raises_value_error(
    translator: PolarsToArcticDBTranslator,
    monkeypatch: MonkeyPatch,
) -> None:
    def raise_syntax_error(_expr: str) -> ast.AST:
        raise SyntaxError("invalid")

    monkeypatch.setattr(translator, "_parse_expression", raise_syntax_error)

    with pytest.raises(ValueError, match="Invalid Polars expression"):
        translator.translate(pl.col("col1"), QueryBuilder())


def test_replace_square_brackets_handles_missing_open_marker(
    translator: PolarsToArcticDBTranslator,
) -> None:
    assert translator._replace_square_brackets("foo])bar") == "foo])bar"


def test_process_node_name_unary_and_unsupported(
    translator: PolarsToArcticDBTranslator,
) -> None:
    assert translator._process_node(ast.parse("field_name", mode="eval").body) == "field_name"
    assert translator._process_node(ast.parse("-1", mode="eval").body) is not None

    with pytest.raises(NotImplementedError, match="Node type List not supported"):
        translator._process_node(ast.parse("[1, 2]", mode="eval").body)


def test_process_call_contains_requires_string_namespace(
    translator: PolarsToArcticDBTranslator,
    monkeypatch: MonkeyPatch,
) -> None:
    call_node = ast.parse('col("a").contains("x")', mode="eval").body
    assert isinstance(call_node, ast.Call)
    assert isinstance(call_node.func, ast.Attribute)
    call_value = call_node.func.value

    original_process_node = translator._process_node

    def fake_process_node(node: ast.AST) -> Any:
        if node is call_node.args[0]:
            return "x"
        if node is call_value:
            return ("lhs", "not_str")
        return original_process_node(node)

    monkeypatch.setattr(translator, "_process_node", fake_process_node)

    with pytest.raises(NotImplementedError, match="Method contains not supported"):
        translator._process_call(call_node)


def test_process_call_unsupported_methods_and_function_shapes(
    translator: PolarsToArcticDBTranslator,
) -> None:
    unsupported_method_node = ast.parse('col("a").foo()', mode="eval").body
    assert isinstance(unsupported_method_node, ast.Call)
    with pytest.raises(NotImplementedError, match="Method foo not supported"):
        translator._process_call(unsupported_method_node)

    unsupported_name_node = ast.parse('foo("a")', mode="eval").body
    assert isinstance(unsupported_name_node, ast.Call)
    assert translator._process_call(unsupported_name_node) is None

    unsupported_func_shape_node = ast.parse("(lambda: 1)()", mode="eval").body
    assert isinstance(unsupported_func_shape_node, ast.Call)
    assert translator._process_call(unsupported_func_shape_node) is None


def test_process_attribute_compare_and_unary_error_branches(
    translator: PolarsToArcticDBTranslator,
) -> None:
    unsupported_attribute_node = ast.parse('col("a").dt', mode="eval").body
    assert isinstance(unsupported_attribute_node, ast.Attribute)
    with pytest.raises(NotImplementedError, match="Attribute dt not supported"):
        translator._process_attribute(unsupported_attribute_node)

    assert (
        translator._process_compare(
            ast.Compare(
                left=ast.Constant("a"),
                ops=[ast.In()],
                comparators=[ast.Constant("b")],
            )
        )
        is not None
    )
    assert (
        translator._process_compare(
            ast.Compare(
                left=ast.Constant("a"),
                ops=[ast.NotIn()],
                comparators=[ast.Constant("b")],
            )
        )
        is not None
    )

    with pytest.raises(NotImplementedError, match="Operator <class 'ast.Is'> not supported"):  # noqa: RUF043
        translator._process_compare(
            ast.Compare(
                left=ast.Constant(1),
                ops=[ast.Is()],
                comparators=[ast.Constant(2)],
            )
        )

    unsupported_unary_node = ast.parse("+1", mode="eval").body
    assert isinstance(unsupported_unary_node, ast.UnaryOp)
    with pytest.raises(NotImplementedError, match="Operator <class 'ast.UAdd'> not supported"):  # noqa: RUF043
        translator._process_unaryop(unsupported_unary_node)


def test_translate_predicate_falls_back_on_unsupported_predicate(
    monkeypatch: MonkeyPatch,
) -> None:
    def raise_not_implemented(_predicate: pl.Expr, _query_builder: QueryBuilder) -> Any:
        raise NotImplementedError("unsupported")

    base_query_builder = QueryBuilder()
    monkeypatch.setattr(polarctic_module._TRANSLATOR, "translate", raise_not_implemented)

    assert (
        polarctic_module._translate_predicate(pl.col("a"), base_query_builder) is base_query_builder
    )
