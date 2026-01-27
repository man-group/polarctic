
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
import os
import pandas as pd
import pandas.testing as pdt
import polars as pl
import pyarrow as pa
import arcticdb
from arcticdb import OutputFormat, VersionedItem, QueryBuilder

import pytest

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

