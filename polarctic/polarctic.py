"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source License, use of
this software will be governed by the Apache License, version 2.0.
"""

import ast
import datetime as dt
import re
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any, cast, overload

import numpy as np
import polars as pl
import pyarrow as pa
from arcticdb import Arctic, LazyDataFrame, OutputFormat, QueryBuilder
from arcticdb.version_store.library import Library, ReadRequest
from arcticdb.version_store.processing import ExpressionNode
from arcticdb_ext.util import RegexGeneric
from arcticdb_ext.version_store import OperationType


class PolarsToArcticDBTranslator:
    """
    Translates Polars expressions to ArcticDB QueryBuilder operations.

    Usage:
        translator = PolarsToArcticDBTranslator()
        qb = translator.translate(polars_expr, query_builder)
    """

    @staticmethod
    @lru_cache(maxsize=512)
    def _parse_expression(expr: str) -> ast.AST:
        return ast.parse(expr, mode="eval").body

    def translate(self, polars_expr: pl.Expr, query_builder: Any) -> Any:
        """
        Translate a Polars expression string to ArcticDB QueryBuilder operations.

        Args:
            polars_expr: String representation of Polars expression
            query_builder: ArcticDB QueryBuilder instance

        Returns:
            Modified QueryBuilder instance
        """

        # Clean the expression - remove surrounding brackets if present
        expr = str(polars_expr).strip()
        if expr.startswith("[") and expr.endswith("]"):
            expr = expr[1:-1].strip()

        expr = self._replace_square_brackets(expr)

        # Preprocess to handle Polars-specific notation like [dyn int: 2]
        expr = self._preprocess_expression(expr)

        # Parse the expression
        try:
            expr_node = self._process_node(self._parse_expression(expr))
        except SyntaxError as e:
            raise ValueError(f"Invalid Polars expression: {polars_expr}") from e

        return query_builder[expr_node]

    def _replace_square_brackets(self, text: str) -> str:
        while True:
            close = text.rfind("])")
            if close == -1:
                break
            open_ = text.rfind("([", 0, close)
            if open_ == -1:
                break
            # Replace the matched ([...]) with (...)
            text = text[:open_] + "(" + text[open_ + 2 : close] + ")" + text[close + 2 :]
        return text

    def _preprocess_expression(self, expr: str) -> str:
        """
        Preprocess Polars expression to handle special notation.

        Converts patterns like [dyn int: 2] to just the value (2).
        """
        # Pattern to match [dyn type: value] or [lit type: value]
        pattern = r"[\[\(](dyn|lit)\s+\w+:\s*([^\]\)]+)[\]\)]"

        def replace_dynamic(match: re.Match) -> Any:
            return f"({match.group(2).strip()})"

        update = re.sub(pattern, replace_dynamic, expr)
        return re.sub(r"\.not\(\)", r".not_()", update)

    def _process_node(self, node: ast.AST) -> Any:
        """Process an AST node and apply corresponding ArcticDB operation."""

        node_type = type(node)
        match node_type:
            case ast.Call:
                return self._process_call(cast(ast.Call, node))
            case ast.Attribute:
                return self._process_attribute(cast(ast.Attribute, node))
            case ast.Name:
                return cast(ast.Name, node).id
            case ast.Constant:
                return cast(ast.Constant, node).value
            case ast.Compare:
                return self._process_compare(cast(ast.Compare, node))
            case ast.BinOp:
                return self._process_binop(cast(ast.BinOp, node))
            case ast.UnaryOp:
                return self._process_unaryop(cast(ast.UnaryOp, node))
            # case ast.List:
            #    return [self._process_node(elt) for elt in node.elts]
            # case ast.Tuple:
            #    return tuple(self._process_node(elt) for elt in node.elts)
            case _:
                raise NotImplementedError(f"Node type {node_type.__name__} not supported")

    def _process_call(self, node: ast.Call) -> Any:
        """Process function calls (e.g., pl.col(), methods)."""

        func_type = type(node.func)
        match func_type:
            case ast.Attribute:
                func = cast(ast.Attribute, node.func)
                attr = func.attr
                match attr:
                    case "negate":
                        new_node = ast.UnaryOp(ast.USub(), func.value)
                        return self._process_unaryop(new_node)
                    case "not_":
                        new_node = ast.UnaryOp(ast.Invert(), func.value)
                        return self._process_unaryop(new_node)
                    case "abs":
                        operand = self._process_node(func.value)
                        return ExpressionNode.compose(operand, OperationType.ABS, None)
                    case "is_null":
                        operand = self._process_node(func.value)
                        return ExpressionNode.compose(operand, OperationType.ISNULL, None)
                    case "contains":
                        arg = self._process_node(node.args[0])
                        left = self._process_node(func.value)
                        if left[1] == "str":
                            return ExpressionNode.compose(
                                left[0], OperationType.REGEX_MATCH, RegexGeneric(arg)
                            )
                        raise NotImplementedError(f"Method {attr} not supported")
                    case "is_in":
                        arg_list = [self._process_node(arg) for arg in node.args]
                        left = self._process_node(func.value)
                        return ExpressionNode.compose(left, OperationType.ISIN, np.array(arg_list))
                    case _:
                        raise NotImplementedError(f"Method {attr} not supported")
            case ast.Name:
                # Function call like col('x')
                func_name = cast(ast.Name, node.func).id
                args = [self._process_node(arg) for arg in node.args]

                if func_name == "col":
                    return ExpressionNode.column_ref(args[0]) if args else None
            case _:
                return None
        return None

    def _process_attribute(self, node: ast.Attribute) -> Any:
        """Process attribute access like pl.col or obj.attr."""
        # TODO rework this
        obj = self._process_node(node.value)
        attr = node.attr

        if attr in ["str"]:
            return (obj, attr)

        raise NotImplementedError(f"Attribute {attr} not supported for object {obj}")

    def _process_compare(self, node: ast.Compare) -> None | ExpressionNode:
        """Process comparison operations and apply filters."""

        left = self._process_node(node.left)

        # Handle multiple comparisons
        expr_node: Any = None
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            right = self._process_node(comparator)
            op_type = type(op)

            match op_type:
                case ast.Eq:
                    expr_node = ExpressionNode.compose(left, OperationType.EQ, right)
                case ast.NotEq:
                    expr_node = ExpressionNode.compose(left, OperationType.NE, right)
                case ast.Lt:
                    expr_node = ExpressionNode.compose(left, OperationType.LT, right)
                case ast.LtE:
                    expr_node = ExpressionNode.compose(left, OperationType.LE, right)
                case ast.Gt:
                    expr_node = ExpressionNode.compose(left, OperationType.GT, right)
                case ast.GtE:
                    expr_node = ExpressionNode.compose(left, OperationType.GE, right)
                case ast.In:
                    expr_node = ExpressionNode.compose(left, OperationType.ISIN, right)
                case ast.NotIn:
                    expr_node = ExpressionNode.compose(left, OperationType.ISNOTIN, right)
                case _:
                    raise NotImplementedError(f"Operator {op_type} not supported")
            left = expr_node  # type: ignore[assignment]

        assert expr_node is not None
        return expr_node

    def _process_binop(self, node: ast.BinOp) -> Any:
        """Process binary operations."""

        left = self._process_node(node.left)

        right = self._process_node(node.right)
        op_type = type(node.op)

        match op_type:
            case ast.Add:
                return ExpressionNode.compose(left, OperationType.ADD, right)
            case ast.Sub:
                return ExpressionNode.compose(left, OperationType.SUB, right)
            case ast.Mult:
                return ExpressionNode.compose(left, OperationType.MUL, right)
            case ast.Div:
                return ExpressionNode.compose(left, OperationType.DIV, right)
            # ArcticDB overloads AND/OR/XOR for both boolean filter composition
            # and integer bitwise expressions, so the same mapping handles both.
            case ast.BitOr:
                return ExpressionNode.compose(left, OperationType.OR, right)
            case ast.BitAnd:
                return ExpressionNode.compose(left, OperationType.AND, right)
            case ast.BitXor:
                return ExpressionNode.compose(left, OperationType.XOR, right)
            case _:
                raise NotImplementedError(f"Operator {op_type} not supported")

    def _process_unaryop(self, node: ast.UnaryOp) -> Any:
        """Process unary operations."""

        operand = self._process_node(node.operand)
        op_type = type(node.op)

        match op_type:
            case ast.Invert:
                return ExpressionNode.compose(operand, OperationType.NOT, None)
            case ast.USub:
                return ExpressionNode.compose(operand, OperationType.NEG, None)
            case _:
                raise NotImplementedError(f"Operator {op_type} not supported")

        return None


def parse_schema(
    lib: Library, symbol: str, as_of: int | str | dt.datetime | None = None
) -> pl.Schema:
    lazy_source = cast(
        LazyDataFrame,
        lib.read(symbol, as_of=as_of, output_format=OutputFormat.PYARROW, lazy=True),
    )
    return cast(pl.Schema, lazy_source._collect_schema())  # type: ignore[attr-defined]


_TRANSLATOR = PolarsToArcticDBTranslator()


@lru_cache(maxsize=32)
def _get_library_from_uri(uri: str, lib_name: str) -> Library:
    return Arctic(uri).get_library(lib_name)


def _translate_predicate(
    predicate: pl.Expr | None,
    query_builder: QueryBuilder | None = None,
) -> QueryBuilder | None:
    if predicate is None:
        return query_builder

    base_query_builder = query_builder or QueryBuilder()
    try:
        return cast(QueryBuilder, _TRANSLATOR.translate(predicate, base_query_builder))
    except (NotImplementedError, ValueError):
        # Unsupported predicate for ArcticDB pushdown; fall back to Polars-side filtering
        return query_builder


def _iter_read_request_batches(
    lib: Library,
    read_request: ReadRequest,
    n_rows: int | None,
    batch_size: int | None,
) -> Iterator[pl.DataFrame]:
    # Fast path: Polars passes batch_size=None for a plain .collect() (no streaming).
    # Execute a single lib.read() round-trip instead of looping with row_range slices.
    if batch_size is None:
        rr = read_request
        if n_rows is not None:
            base_start = 0
            if rr.row_range is not None and rr.row_range[0] is not None:
                base_start = rr.row_range[0]
            end = base_start + n_rows
            if rr.row_range is not None and rr.row_range[1] is not None:
                end = min(end, rr.row_range[1])
            rr = rr._replace(row_range=(base_start, end))
        arrow_table = cast(pa.Table, lib.read(**rr._asdict()).data)
        if arrow_table.num_rows > 0:
            yield cast(pl.DataFrame, pl.from_arrow(arrow_table, rechunk=False))
        return

    # Streaming path: yield fixed-size batches so Polars can process them incrementally.
    # ArcticDB LazyDataFrame.row_range mutates in place, so each batch must use a
    # fresh ReadRequest with an absolute row_range.
    effective_batch_size = batch_size
    base_start = 0
    base_end: int | None = None
    if read_request.row_range is not None:
        start, end = read_request.row_range
        if start is not None:
            base_start = start
        if end is not None:
            base_end = end

    read_offset = 0
    remaining_rows = n_rows

    while remaining_rows is None or remaining_rows > 0:
        current_batch_size = (
            effective_batch_size
            if remaining_rows is None
            else min(effective_batch_size, remaining_rows)
        )

        batch_start = base_start + read_offset
        batch_end = batch_start + current_batch_size
        if base_end is not None:
            batch_end = min(batch_end, base_end)
        if batch_end <= batch_start:
            break

        batch_request = read_request._replace(row_range=(batch_start, batch_end))
        arrow_table = cast(pa.Table, lib.read(**batch_request._asdict()).data)
        rows_read = arrow_table.num_rows

        if rows_read == 0:
            break

        yield cast(pl.DataFrame, pl.from_arrow(arrow_table, rechunk=False))

        read_offset += rows_read
        if remaining_rows is not None:
            remaining_rows -= rows_read
        if rows_read < current_batch_size:
            break


def _register_arctic_source(
    lib: Library,
    schema_getter: Callable[[], pl.Schema],
    read_request_getter: Callable[[], ReadRequest],
) -> pl.LazyFrame:
    # Cache the schema: Polars may call the getter repeatedly during lazy plan
    # construction (after each .filter(), .select(), etc.).  The schema of a
    # versioned symbol is immutable, so one call is always sufficient.
    _cached_schema: pl.Schema | None = None

    def get_schema() -> pl.Schema:
        nonlocal _cached_schema
        if _cached_schema is None:
            _cached_schema = schema_getter()
        return _cached_schema

    _cached_read_request: ReadRequest | None = None

    def get_base_read_request() -> ReadRequest:
        nonlocal _cached_read_request
        if _cached_read_request is None:
            base = read_request_getter()
            if base.output_format != OutputFormat.PYARROW:
                base = base._replace(output_format=OutputFormat.PYARROW)
            _cached_read_request = base
        return _cached_read_request

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        read_request = get_base_read_request()

        if with_columns is not None:
            read_request = read_request._replace(columns=with_columns)

        translated_predicate = _translate_predicate(predicate, read_request.query_builder)
        if translated_predicate is not read_request.query_builder:
            read_request = read_request._replace(query_builder=translated_predicate)

        yield from _iter_read_request_batches(lib, read_request, n_rows, batch_size)

    return pl.io.plugins.register_io_source(  # type: ignore[attr-defined]
        io_source=source_generator,
        schema=get_schema,
    )


def _scan_lazy_dataframe(source: LazyDataFrame) -> pl.LazyFrame:
    """Register a Polars IO source backed by an existing ArcticDB LazyDataFrame.

    The LazyDataFrame may already carry ArcticDB-level QueryBuilder operations
    (projections, filters). Any additional Polars predicates or
    column selections are pushed down on top of those.
    """

    return _register_arctic_source(
        lib=cast(Library, source.lib),
        schema_getter=lambda: cast(pl.Schema, source._collect_schema()),  # type: ignore[attr-defined]
        read_request_getter=lambda: cast(ReadRequest, source._to_read_request()),  # type: ignore[attr-defined]
    )


@overload
def scan_arcticdb(
    source: str,
    lib_name: str,
    symbol: str,
    /,
    *,
    as_of: int | str | dt.datetime | None = None,
) -> pl.LazyFrame: ...


@overload
def scan_arcticdb(
    source: Library,
    symbol: str,
    /,
    *,
    as_of: int | str | dt.datetime | None = None,
) -> pl.LazyFrame: ...


@overload
def scan_arcticdb(
    source: LazyDataFrame,
    /,
) -> pl.LazyFrame: ...


def scan_arcticdb(
    source: str | Library | LazyDataFrame,
    lib_name_or_symbol: str | None = None,
    symbol: str | None = None,
    /,
    *,
    as_of: int | str | dt.datetime | None = None,
) -> pl.LazyFrame:
    """
    Create a Polars LazyFrame backed by an ArcticDB symbol.

    Three calling forms are supported:

    1. URI form (highest overhead — opens an Arctic connection on each call)::
           scan_arcticdb(uri, lib_name, symbol, *, as_of=None)

    2. Library form (preferred for repeated calls against the same library)::
           scan_arcticdb(lib, symbol, *, as_of=None)

    3. LazyDataFrame form (pre-apply ArcticDB operations before Polars sees the data)::
           scan_arcticdb(lazy_df)
    """
    if isinstance(source, str):
        if lib_name_or_symbol is None or symbol is None:
            raise ValueError("lib_name and symbol are required when source is a URI string")
        lib = _get_library_from_uri(source, lib_name_or_symbol)
    elif isinstance(source, Library):
        lib = source
        if lib_name_or_symbol is None:
            raise ValueError("symbol is required when source is a Library")
        symbol = lib_name_or_symbol
    elif isinstance(source, LazyDataFrame):
        return _scan_lazy_dataframe(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")

    base_lazy_source = cast(
        LazyDataFrame,
        lib.read(
            symbol,
            as_of=as_of,
            lazy=True,
            output_format=OutputFormat.PYARROW,
        ),
    )

    return _register_arctic_source(
        lib=lib,
        # _collect_schema() reads the schema from symbol metadata (no data IO),
        # which is faster than parse_schema()'s row_range=(0,1) data read.
        schema_getter=lambda: cast(pl.Schema, base_lazy_source._collect_schema()),  # type: ignore[attr-defined]
        read_request_getter=lambda: cast(
            ReadRequest,
            base_lazy_source._to_read_request(),  # type: ignore[attr-defined]
        ),
    )
