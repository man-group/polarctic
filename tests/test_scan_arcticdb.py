from typing import Any

import pandas as pd
import pandas.testing as pdt
import polars as pl
import pytest
from arcticdb import OutputFormat, QueryBuilder, VersionedItem

import polarctic.polarctic as polarctic_module

"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source License, use of
this software will be governed by the Apache License, version 2.0.
"""

"""
Unit tests for polarctic.scan_arcticdb using a real ArcticDB LMDB-backed store.

Dependencies (imported at module import time, not lazily):
- pandas
- polars
- pyarrow
- arcticdb

Each test takes both fixtures init_arcticdb and delete_arcticdb so setup runs
before the test and teardown removes the LMDB store afterwards.
"""


def test_parse_schema_returns_expected_schema(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    """
    Test polarctic.parse_schema by passing the real Library instance and ensuring
    the returned polars.Schema lists the columns we wrote.
    """
    lib = init_arcticdb["lib"]
    symbol = "df1"

    schema = polarctic_module.parse_schema(lib, symbol)
    assert isinstance(schema, pl.Schema)

    # our fixture wrote columns 'a' and 'ts' for df1
    assert "a" in schema.names()
    assert "ts" in schema.names()


def test_scan_arcticdb_reads_data(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    info = init_arcticdb
    uri = info["uri"]
    lib_name = info["lib_name"]
    expected_tables: dict = info["tables"]

    for symbol, expected_df in expected_tables.items():
        lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, symbol)
        pl_df: pl.DataFrame = lazy.collect()
        pd_df: pd.DataFrame = pl_df.to_pandas()
        pdt.assert_frame_equal(pd_df, expected_df, check_dtype=False, check_like=True)


def test_scan_arcticdb_with_filter(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    info = init_arcticdb
    uri = info["uri"]
    lib = info["lib"]
    lib_name = info["lib_name"]
    expected_tables: dict = info["tables"]

    filters = [pl.col("a") > 4, pl.col("b") < 19, ((pl.col("a") > 4) & (pl.col("b") < 19))]
    qe1 = QueryBuilder()
    qe1 = qe1[qe1["a"] > 4]
    qe2 = QueryBuilder()
    qe2 = qe2[qe2["b"] < 19]
    qe3 = QueryBuilder()
    qe3 = qe3[(qe3["a"] > 4) & (qe3["b"] < 19)]
    query_builders = [qe1, qe2, qe3]

    for filter, query_builder in zip(filters, query_builders, strict=False):
        for symbol in expected_tables:
            lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, symbol)
            pl_df: pl.DataFrame = lazy.filter(filter).collect()
            pd_df: pd.DataFrame = pl_df.to_pandas()
            arctic_df: VersionedItem = lib.read(
                symbol, query_builder=query_builder, output_format=OutputFormat.PANDAS
            )
            pdt.assert_frame_equal(pd_df, arctic_df.data, check_dtype=False, check_like=True)


def test_scan_arcticdb_with_select(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    info = init_arcticdb
    uri = info["uri"]
    lib_name = info["lib_name"]
    expected_tables: dict = info["tables"]

    for symbol, expected_df in expected_tables.items():
        lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, symbol)
        pl_df: pl.DataFrame = lazy.select(pl.col("a"), pl.col("b")).collect()
        pd_df: pd.DataFrame = pl_df.to_pandas()
        pdt.assert_frame_equal(pd_df, expected_df[["a", "b"]], check_dtype=False, check_like=True)


def test_scan_arcticdb_library_source(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    """scan_arcticdb(lib, symbol) form produces the same result as the URI form."""
    info = init_arcticdb
    uri = info["uri"]
    lib = info["lib"]
    lib_name = info["lib_name"]
    expected_tables: dict = info["tables"]

    for symbol, _expected_df in expected_tables.items():
        lazy_lib: pl.LazyFrame = polarctic_module.scan_arcticdb(lib, symbol)
        lazy_uri: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, symbol)
        pdt.assert_frame_equal(
            lazy_lib.collect().to_pandas(),
            lazy_uri.collect().to_pandas(),
            check_dtype=False,
            check_like=True,
        )


def test_scan_arcticdb_lazy_dataframe_source(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    """scan_arcticdb(lazy_df) reads the same data as the Library form."""
    info = init_arcticdb
    lib = info["lib"]
    lib_name = info["lib_name"]
    uri = info["uri"]
    expected_tables: dict = info["tables"]

    for symbol in expected_tables:
        lazy_df = lib.read(symbol, lazy=True, output_format=OutputFormat.PYARROW)
        lf: pl.LazyFrame = polarctic_module.scan_arcticdb(lazy_df)
        pdt.assert_frame_equal(
            lf.collect().to_pandas(),
            polarctic_module.scan_arcticdb(uri, lib_name, symbol).collect().to_pandas(),
            check_dtype=False,
            check_like=True,
        )


def test_scan_arcticdb_lazy_dataframe_with_prefilter(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    """ArcticDB-level QB pre-filter on the LazyDataFrame is preserved, and an
    additional Polars predicate is pushed down on top of it."""
    info = init_arcticdb
    lib = info["lib"]
    symbol = "df1"

    # Pre-filter via ArcticDB QB: a > 4 (rows 5-9)
    qb = QueryBuilder()
    qb = qb[qb["a"] > 4]
    lazy_df = lib.read(symbol, query_builder=qb, lazy=True, output_format=OutputFormat.PYARROW)

    # Additional Polars predicate: b < 19 (keeps rows where b < 19)
    lf: pl.LazyFrame = polarctic_module.scan_arcticdb(lazy_df)
    result = lf.filter(pl.col("b") < 19).collect().to_pandas()

    # Expected: rows where a > 4 AND b < 19
    combined_qb = QueryBuilder()
    combined_qb = combined_qb[(combined_qb["a"] > 4) & (combined_qb["b"] < 19)]
    expected = lib.read(symbol, query_builder=combined_qb, output_format=OutputFormat.PANDAS).data

    pdt.assert_frame_equal(result, expected, check_dtype=False, check_like=True)


def test_scan_arcticdb_lazy_dataframe_with_column_selection(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    """Column selection (select) is pushed down into the LazyDataFrame read."""
    info = init_arcticdb
    lib = info["lib"]
    symbol = "df1"

    lazy_df = lib.read(symbol, lazy=True, output_format=OutputFormat.PYARROW)
    lf: pl.LazyFrame = polarctic_module.scan_arcticdb(lazy_df)
    result = lf.select(pl.col("a"), pl.col("b")).collect().to_pandas()

    expected = lib.read(symbol, columns=["a", "b"], output_format=OutputFormat.PANDAS).data
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_like=True)


def test_scan_arcticdb_lazy_dataframe_with_row_range(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    """A pre-sliced LazyDataFrame keeps its base row_range when scanned."""
    info = init_arcticdb
    lib = info["lib"]
    symbol = "df1"

    lazy_df = lib.read(
        symbol,
        row_range=(2, 7),
        lazy=True,
        output_format=OutputFormat.PYARROW,
    )
    lf: pl.LazyFrame = polarctic_module.scan_arcticdb(lazy_df)
    result = lf.collect().to_pandas()

    expected = lib.read(
        symbol,
        row_range=(2, 7),
        output_format=OutputFormat.PANDAS,
    ).data
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_like=True)


def test_scan_arcticdb_lazy_dataframe_with_schema_changing_projection(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    """Schema-changing ArcticDB preprocessing is reflected in Polars schema and data."""
    info = init_arcticdb
    lib = info["lib"]
    symbol = "df1"
    expected = info["tables"][symbol].copy()

    lazy_df = lib.read(symbol, lazy=True, output_format=OutputFormat.PYARROW)
    lazy_df["c"] = lazy_df["a"] + lazy_df["b"]

    lf: pl.LazyFrame = polarctic_module.scan_arcticdb(lazy_df)
    schema = lf.collect_schema()

    assert "c" in schema.names()
    assert schema["c"] == pl.Float64

    result = lf.collect().to_pandas()
    expected["c"] = expected["a"] + expected["b"]
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_like=True)


def test_iter_read_request_batches_fast_path_respects_row_range_and_n_rows(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    del delete_arcticdb
    lib = init_arcticdb["lib"]
    lazy_df = lib.read("df1", lazy=True, output_format=OutputFormat.PYARROW)
    read_request = lazy_df._to_read_request()._replace(row_range=(2, 8))

    batches = list(
        polarctic_module._iter_read_request_batches(
            lib,
            read_request,
            n_rows=3,
            batch_size=None,
        )
    )

    assert [batch.height for batch in batches] == [3]
    assert batches[0]["a"].to_list() == [2, 3, 4]


def test_iter_read_request_batches_streaming_handles_n_rows_limit(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    del delete_arcticdb
    lib = init_arcticdb["lib"]
    lazy_df = lib.read("df1", lazy=True, output_format=OutputFormat.PYARROW)
    read_request = lazy_df._to_read_request()

    batches = list(
        polarctic_module._iter_read_request_batches(
            lib,
            read_request,
            n_rows=3,
            batch_size=2,
        )
    )

    assert [batch.height for batch in batches] == [2]


def test_iter_read_request_batches_streaming_stops_on_empty_batch(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    del delete_arcticdb
    lib = init_arcticdb["lib"]
    lazy_df = lib.read("df1", lazy=True, output_format=OutputFormat.PYARROW)
    read_request = lazy_df._to_read_request()._replace(row_range=(20, 25))

    batches = list(
        polarctic_module._iter_read_request_batches(
            lib,
            read_request,
            n_rows=4,
            batch_size=2,
        )
    )

    assert batches == []


def test_register_arctic_source_normalizes_output_format(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    del delete_arcticdb
    lib = init_arcticdb["lib"]
    symbol = "df1"
    expected = init_arcticdb["tables"][symbol]

    lazy_df = lib.read(symbol, lazy=True, output_format=OutputFormat.PYARROW)
    read_request = lazy_df._to_read_request()._replace(output_format=OutputFormat.PANDAS)

    lf = polarctic_module._register_arctic_source(
        lib=lib,
        schema_getter=lambda: polarctic_module.parse_schema(lib, symbol),
        read_request_getter=lambda: read_request,
    )

    result = lf.collect().to_pandas()
    pdt.assert_frame_equal(result, expected, check_dtype=False, check_like=True)


def test_scan_arcticdb_validates_input_combinations(
    init_arcticdb: dict[str, Any],
    delete_arcticdb: Any,
) -> None:
    del delete_arcticdb
    uri = init_arcticdb["uri"]
    lib_name = init_arcticdb["lib_name"]
    lib = init_arcticdb["lib"]

    with pytest.raises(ValueError, match="lib_name and symbol are required"):
        polarctic_module.scan_arcticdb(uri, lib_name)

    with pytest.raises(ValueError, match="symbol is required when source is a Library"):
        polarctic_module.scan_arcticdb(lib)

    with pytest.raises(TypeError, match="Unsupported source type"):
        polarctic_module.scan_arcticdb(123)  # type: ignore[arg-type]
