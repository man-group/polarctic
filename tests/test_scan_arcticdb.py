
"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source License, use of this software will be governed by the Apache License, version 2.0.
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
import sys
from unittest.mock import patch

import pandas as pd
import pandas.testing as pdt
import polars as pl
import pytest
from arcticdb import OutputFormat, QueryBuilder, VersionedItem

import polarctic.polarctic as polarctic_module


def test_parse_schema_returns_expected_schema(init_arcticdb, delete_arcticdb):
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


def test_scan_arcticdb_reads_data(init_arcticdb, delete_arcticdb):
    info = init_arcticdb
    uri = info["uri"]
    lib_name = info["lib_name"]
    expected_tables: dict = info["tables"]

    for symbol, expected_df in expected_tables.items():
        lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, symbol)
        pl_df: pl.DataFrame = lazy.collect()
        pd_df: pd.DataFrame = pl_df.to_pandas()
        pdt.assert_frame_equal(pd_df, expected_df, check_dtype=False, check_like=True)

def test_scan_articdb_with_filter(init_arcticdb, delete_arcticdb):
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

    for filter, query_builder in zip(filters, query_builders):
        for symbol in expected_tables:
            lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, symbol)
            pl_df: pl.DataFrame = lazy.filter(filter).collect()
            pd_df: pd.DataFrame = pl_df.to_pandas()
            arctic_df: VersionedItem = lib.read(symbol, query_builder = query_builder, output_format = OutputFormat.PANDAS)
            pdt.assert_frame_equal(pd_df, arctic_df.data, check_dtype=False, check_like=True)

def test_scan_arcticdb_with_select(init_arcticdb, delete_arcticdb):
    info = init_arcticdb
    uri = info["uri"]
    lib = info["lib"]
    lib_name = info["lib_name"]
    expected_tables: dict = info["tables"]

    for symbol, expected_df in expected_tables.items():
        lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, symbol)
        pl_df: pl.DataFrame = lazy.select(pl.col("a"), pl.col("b")).collect()
        pd_df: pd.DataFrame = pl_df.to_pandas()
        pdt.assert_frame_equal(pd_df, expected_df[["a", "b"]], check_dtype=False, check_like=True)


def test_scan_arcticdb_unsupported_predicate_falls_back_to_polars(init_arcticdb, delete_arcticdb):
    """
    When the predicate cannot be translated to ArcticDB (e.g. modulo operator),
    scan_arcticdb must NOT raise; instead it falls back to Polars-side filtering
    and still returns the correct rows.
    """
    info = init_arcticdb
    uri = info["uri"]
    lib_name = info["lib_name"]
    expected_tables: dict = info["tables"]

    # Confirm the predicate cannot be translated before relying on the fallback.
    predicate = pl.col("a") % 2 == 0
    with pytest.raises(NotImplementedError):
        polarctic_module.PolarsToArcticDBTranslator().translate(predicate, QueryBuilder())

    for symbol, expected_df in expected_tables.items():
        lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, symbol)
        pl_df: pl.DataFrame = lazy.filter(predicate).collect()
        pd_df: pd.DataFrame = pl_df.to_pandas()

        expected = expected_df[expected_df["a"] % 2 == 0].reset_index(drop=True)
        pdt.assert_frame_equal(pd_df, expected, check_dtype=False, check_like=True)


def test_scan_arcticdb_unsupported_predicate_skips_arcticdb_pushdown(init_arcticdb, delete_arcticdb):
    """
    When the predicate cannot be translated, query_builder passed to lib.read
    must be None (no pushdown attempted), and the Polars filter is applied instead.
    """
    info = init_arcticdb
    uri = info["uri"]
    lib_name = info["lib_name"]

    predicate = pl.col("a") % 2 == 0
    # Confirm the predicate cannot be translated before relying on the fallback.
    with pytest.raises(NotImplementedError):
        polarctic_module.PolarsToArcticDBTranslator().translate(predicate, QueryBuilder())

    with patch.object(polarctic_module.PolarsToArcticDBTranslator, "translate",
                      side_effect=NotImplementedError("modulo not supported")) as mock_translate:
        lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, "df1")
        pl_df: pl.DataFrame = lazy.filter(predicate).collect()

    # Translation was attempted exactly once
    mock_translate.assert_called_once()

    # Results are still correct (Polars fallback applied)
    expected = info["tables"]["df1"]
    expected = expected[expected["a"] % 2 == 0].reset_index(drop=True)
    pdt.assert_frame_equal(pl_df.to_pandas(), expected, check_dtype=False, check_like=True)


def test_scan_arcticdb_unsupported_predicate_combined_with_column_select(init_arcticdb, delete_arcticdb):
    """
    Unsupported predicate fallback must compose correctly with column projection.
    """
    info = init_arcticdb
    uri = info["uri"]
    lib_name = info["lib_name"]

    predicate = pl.col("a") % 2 == 0
    # Confirm the predicate cannot be translated before relying on the fallback.
    with pytest.raises(NotImplementedError):
        polarctic_module.PolarsToArcticDBTranslator().translate(predicate, QueryBuilder())

    lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, "df1")
    pl_df: pl.DataFrame = lazy.filter(predicate).select("a", "b").collect()

    expected = info["tables"]["df1"]
    expected = expected[expected["a"] % 2 == 0][["a", "b"]].reset_index(drop=True)
    pdt.assert_frame_equal(pl_df.to_pandas(), expected, check_dtype=False, check_like=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only LMDB lock-file stress test")
@pytest.mark.parametrize("iteration", range(8))
def test_scan_arcticdb_windows_cleanup_stress(init_arcticdb, delete_arcticdb, iteration):
    """
    Stress fixture setup/teardown on Windows to catch intermittent LMDB lock-file
    cleanup issues in CI.
    """
    info = init_arcticdb
    uri = info["uri"]
    lib_name = info["lib_name"]
    expected = info["tables"]["df1"]

    predicate = pl.col("a") % 2 == 0
    lazy: pl.LazyFrame = polarctic_module.scan_arcticdb(uri, lib_name, "df1")
    pd_df: pd.DataFrame = lazy.filter(predicate).collect().to_pandas()

    expected = expected[expected["a"] % 2 == 0].reset_index(drop=True)
    pdt.assert_frame_equal(pd_df, expected, check_dtype=False, check_like=True)
