"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source License, use of
this software will be governed by the Apache License, version 2.0.
"""

import ast
import copy
import datetime as dt
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Any, cast, overload

import numpy as np
import polars as pl
from arcticdb import Arctic, LazyDataFrame, OutputFormat, QueryBuilder
from arcticdb.version_store.library import Library, ReadRequest
from arcticdb.version_store.processing import ExpressionNode, PythonFilterClause
from arcticdb_ext.util import RegexGeneric
from arcticdb_ext.version_store import OperationType, PreloadedIndexQuery
from arcticdb_ext.storage import KeyType

from polarctic.streaming import plan_row_reads


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
            # Map these operators to ArcticDB expression nodes.
            # Integer bitwise queries can be constructed, but execution is not
            # currently supported, as documented by the execution tests
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
        lib.read(symbol, as_of=as_of, output_format=OutputFormat.POLARS, lazy=True),
    )
    return lazy_source._collect_schema()


_TRANSLATOR = PolarsToArcticDBTranslator()


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
        frame = cast(pl.DataFrame, lib.read(**rr._asdict()).data)
        if frame.height > 0:
            yield frame
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
        frame = cast(pl.DataFrame, lib.read(**batch_request._asdict()).data)
        rows_read = frame.height

        if rows_read == 0:
            break

        yield frame

        read_offset += rows_read
        if remaining_rows is not None:
            remaining_rows -= rows_read
        if rows_read < current_batch_size:
            break


def _register_arctic_source(source: LazyDataFrame) -> pl.LazyFrame:
    """Register a version-pinned scan and group eligible reads by stored row slices.

    Load ArcticDB's table information, including the selected version and index.
    Build the schema from that loaded index.
    Fetch the stored row ranges separately because they are not exposed directly.
    During collection, reuse the index and read nonoverlapping row ranges.
    """

    # Create a copy to pin the scan to one version but not modify the original query passed to scan_arcticdb()
    source = copy.copy(source)

    cached_schema: pl.Schema | None = None
    cached_slices: list[tuple[int, int]] | None = None
    resolved_version: int | None = None

    def get_schema() -> pl.Schema:
        """Load the selected index and build its schema once."""
        nonlocal cached_schema, resolved_version

        if cached_schema is None:
            if source._preloaded_index is None:
                # Choose the requested version and load its index together
                info = source.lib._nvs._get_info(
                    source.read_request.symbol,
                    source.read_request.as_of,
                    iterate_snapshots_if_tombstoned=False,
                    include_index_segment=True,
                )

                # Determine the selected version from the index key
                resolved_version = info.key.version_id

                # Keep that exact version in our copied read settings
                source.read_request = source.read_request._replace(
                    as_of=resolved_version,
                )

                # Save the index just loaded so schema building and later data reads can reuse it
                source._preloaded_index = PreloadedIndexQuery(
                    info.key,
                    info.index_segment,
                    info.column_stats_segment,
                )

            else:
                # Keep the preloaded index (we need its version number to fetch the matching row layout).
                # Requesting zero rows using that exact index tells us which version the index belongs to.
                # In the tested build, this fetched no additional data or index keys:
                version_probe = source.lib.read(
                    source.read_request.symbol,
                    as_of=source._preloaded_index,
                    row_range=(0, 0),
                    columns=[],
                    output_format=OutputFormat.POLARS,
                )

                resolved_version = version_probe.version

                # Save the version number on the copied wrapper
                source.read_request = source.read_request._replace(
                    as_of=resolved_version,
                )

            assert source._preloaded_index is not None, (
                "Index resolution did not provide a preloaded index"
            )

            # Keep any processing already attached to the ArcticDB query
            request = source._to_read_request()

            # Get column names/types from the loaded index but do not
            # call _collect_schema() as it would fetch the index again
            cached_schema = source.lib._nvs._modify_schema(
                source._preloaded_index,
                columns=request.columns,
                query_builder=request.query_builder,
                arrow_string_format_default=request.arrow_string_format_default,
                arrow_string_format_per_column=request.arrow_string_format_per_column,
            )

            # Allow filters or a plain reads with no processing steps for now
            clauses = request.query_builder._python_clauses
            filters_only = all(
                isinstance(clause, PythonFilterClause)
                for clause in clauses
            )

            if (
                resolved_version is not None
                and filters_only
                and request.date_range is None  # date-range reads are not handled by the new planner yet
            ):
                # Load the stored row boundaries once and save them in cached_slices.
                # Later, collection can use them without another layout fetch per batch
                get_row_slices()

        return cached_schema

    def get_row_slices() -> list[tuple[int, int]]:
        """Read and cache unique stored row ranges for the selected version."""
        nonlocal cached_slices

        if cached_slices is None:
            assert resolved_version is not None

            # Read the layout once for the exact version selected above,
            # then cache the extracted ranges so we do not repeat this for each batch
            layout = source.lib._nvs.read_index(
                source.read_request.symbol,
                as_of=resolved_version,
            )

            # Keep entries describing stored table-data segments
            data_layout = layout[
                layout["key_type"] == int(KeyType.TABLE_DATA)
            ]

            # Several column segments may cover the same source rows.
            # Keep each row range once, then sort the ranges by position
            cached_slices = sorted(
                {
                    (int(start), int(end))
                    for start, end in zip(
                        data_layout["start_row"],
                        data_layout["end_row"],
                        strict=True,
                    )
                }
            )

        return cached_slices

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """Read planned source ranges, or use the existing batch reader."""

        # Resolve the schema/index
        get_schema()

        # Build a fresh request containing the source's existing processing
        read_request = source._to_read_request()

        # Match ArcticDB's LazyDataFrame.collect(): reuse the resolved index
        read_request = read_request._replace(
            as_of=source._preloaded_index,
            output_format=OutputFormat.POLARS,
        )

        if with_columns is not None:
            read_request = read_request._replace(columns=with_columns)

        # Add the new filter to the existing backend query (without removing existing backend filters)
        translated_predicate = _translate_predicate(predicate, read_request.query_builder)

        # The original builder is returned on failure so check this before replacing the builder in the request
        translated_ok = (predicate is None or translated_predicate is not read_request.query_builder)

        if translated_predicate is not read_request.query_builder:
            read_request = read_request._replace(
                query_builder=translated_predicate,
            )

        # Inspect the processing steps in the completed backend query
        query_builder = read_request.query_builder
        clauses = (
            []
            if query_builder is None
            else query_builder._python_clauses
        )

        # Filters can run separately on each group of rows.
        # Other processing, such as aggregation, is not handled yet
        filters_only = all(
            isinstance(clause, PythonFilterClause)
            for clause in clauses
        )
        has_filter = bool(clauses) or predicate is not None

        # Handle positions counted from the start
        row_range = read_request.row_range or (None, None)
        nonnegative_bounds = all(
            position is None or position >= 0
            for position in row_range
        )

        # Check whether this request can use stored-slice planning
        aligned = (
            batch_size is not None
            and batch_size > 0
            and translated_ok
            and filters_only
            # Filter-before-limit and limit-before-filter differ.
            # Keep this combination on the old reader for now
            and not (has_filter and n_rows is not None)
            and read_request.date_range is None
            and resolved_version is not None
            and nonnegative_bounds
        )

        if aligned:
            # Group stored slices using Polars' batch hint.
            # Each tuple describes one backend read
            ranges = plan_row_reads(
                get_row_slices(),
                batch_size,
                read_request.row_range,
                n_rows,
            )

            for start, end in ranges:
                # Read this planned range using the same loaded index
                batch_request = read_request._replace(
                    row_range=(start, end),
                )
                frame = source.lib.read(
                    **batch_request._asdict()
                ).data

                # Native filtering may leave no matching rows, so send only nonempty frames to Polars
                if frame.height > 0:
                    yield frame

                # Continue to the next planned range even if nothing matched.
                # Do not calculate the next position from frame.height

            return

        # Use the existing reader for requests not handled by the new planner
        yield from _iter_read_request_batches(
            source.lib,
            read_request,
            n_rows,
            batch_size,
        )

    return pl.io.plugins.register_io_source(
        io_source=source_generator,
        schema=get_schema,
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
        return _register_arctic_source(source)
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")

    base_lazy_source = cast(
        LazyDataFrame,
        lib.read(
            symbol,
            as_of=as_of,
            lazy=True,
            output_format=OutputFormat.POLARS,
        ),
    )

    # All entry points use the same schema/index handling
    return _register_arctic_source(base_lazy_source)
