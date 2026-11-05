"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

import copy
import gc
import datetime as dt
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from arcticdb import Arctic, LibraryOptions, OutputFormat
from arcticdb.version_store.library import Library

import polarctic.polarctic as polarctic_module
from polarctic.polarctic import PolarsToArcticDBTranslator


@pytest.fixture
def translator() -> PolarsToArcticDBTranslator:
    """Create a new translator for the test."""
    return PolarsToArcticDBTranslator()


@pytest.fixture
def init_arcticdb(tmp_path: Path) -> dict[str, Any]:
    """
    Initialize a real ArcticDB LMDB-backed store and write two pandas DataFrames
    into a library.

    Returns a dict with:
      - uri: the URI passed to Arctic(...) (string)
      - lmdb_path: path to LMDB dir that will be removed in teardown
      - lib_name: the created library name (string)
      - tables: a dict of the pandas DataFrames written, keyed by symbol name
      - ac: the Arctic instance
      - lib: the Library instance
    """
    # Prepare LMDB directory
    lmdb_dir = Path(tmp_path) / "arctic_lmdb"
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = str(lmdb_dir)

    uri = f"lmdb://{lmdb_path}"
    lib_name = "test_lib"

    # Instantiate Arctic and create library
    ac = Arctic(uri)
    lib = ac.create_library(lib_name)

    # Prepare pandas DataFrames to write
    df1 = pd.DataFrame(
        {
            "a": np.arange(10, dtype=np.int64),
            "b": np.arange(10, 20, dtype=np.float64),
            "ts": pd.date_range("2020-01-01", periods=10),
        }
    )
    df2 = pd.DataFrame(
        {
            "a": np.arange(2, 12, dtype=np.float64),
            "b": np.arange(15, 25, dtype=np.int64),
            "ts": pd.date_range("2020-02-01", periods=10),
        }
    )

    # Write DataFrames into the library under symbols "df1" and "df2".
    lib.write("df1", df1)
    lib.write("df2", df2)

    return {
        "uri": uri,
        "lmdb_path": lmdb_path,
        "lib_name": lib_name,
        "tables": {"df1": df1, "df2": df2},
        "ac": ac,
        "lib": lib,
    }


@pytest.fixture
def delete_arcticdb(init_arcticdb: dict[str, Any]) -> Iterator[None]:
    """
    Teardown fixture: remove the LMDB directory created by init_arcticdb.
    Tests should include this fixture in the parameter list to ensure cleanup.
    """
    yield
    lib_name = init_arcticdb["lib_name"]
    lmdb_path = init_arcticdb["lmdb_path"]
    ac = init_arcticdb["ac"]

    # Drop strong references held by the fixture dict before deleting on-disk state.
    init_arcticdb.pop("lib", None)
    init_arcticdb.pop("tables", None)
    gc.collect()

    # Windows may keep LMDB lock files briefly; retry deletion a few times.
    for _ in range(10):
        try:
            ac.delete_library(lib_name)
            break
        except Exception:
            gc.collect()
            time.sleep(0.1)

    init_arcticdb.pop("ac", None)
    del ac
    gc.collect()

    for _ in range(10):
        try:
            shutil.rmtree(lmdb_path)
            break
        except Exception:
            gc.collect()
            time.sleep(0.1)


@dataclass
class ScanStore:
    """Record the ArcticDB test database details and expected Polars table."""
    uri: str
    library_name: str
    library: Library
    expected: pl.DataFrame


@dataclass
class ObservedRead:
    """Record the read options and the table returned by ArcticDB."""
    kwargs: dict[str, Any]
    data: pa.Table


@pytest.fixture
def scan_store(tmp_path: Path) -> Iterator[ScanStore]:
    """Create a small test database and independent expected Polars data for one test."""
    uri = f"lmdb://{tmp_path / 'connector_store'}"
    ac = Arctic(uri)
    lib = ac.create_library(
        "connector_tests",
        library_options=LibraryOptions(rows_per_segment=3, columns_per_segment=2, dynamic_schema=True),
    )
    expected = pl.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, -5, -2],
            "b": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 8.0, 9.0],
            "desk": ["rates", "fx", "rates", "fx", "credit", "credit", "rates", "fx"],
            "flag": [True, False, True, False, True, False, True, False],
            "text": ["alpha", 'a"b', "[x]", "plain", "beta", "x", "negative", "minus"],
            "ts": pl.datetime_range(
                dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 8), interval="1d", time_unit="ns", eager=True
            ),
        },
        schema_overrides={"a": pl.Int64, "b": pl.Float64},
    )
    # pandas is only used to write fixtures; all result comparisons stay in Polars
    lib.write("data", expected.to_pandas())
    yield ScanStore(uri, "connector_tests", lib, expected)
    ac.delete_library("connector_tests")


@pytest.fixture(params=["uri", "library"])
def scan_factory(request: pytest.FixtureRequest, scan_store: ScanStore, monkeypatch: pytest.MonkeyPatch) -> Callable[..., pl.LazyFrame]:
    """Build scans using the URI and existing-library calling forms."""
    if request.param == "uri":
        # Reuse the open library; real URI opening is tested separately
        def resolve(uri: str, library_name: str) -> Library:
            assert uri == scan_store.uri and library_name == scan_store.library_name
            return scan_store.library

        monkeypatch.setattr(polarctic_module, "_get_library_from_uri", resolve)

    def scan(symbol: str = "data", *, as_of: int | str | dt.datetime | None = 0) -> pl.LazyFrame:
        if request.param == "uri":
            return polarctic_module.scan_arcticdb(scan_store.uri, scan_store.library_name, symbol, as_of=as_of)
        
        return polarctic_module.scan_arcticdb(scan_store.library, symbol, as_of=as_of)

    return scan


@pytest.fixture
def observed_reads(monkeypatch: pytest.MonkeyPatch) -> list[ObservedRead]:
    """Record the arguments and Arrow output of actual ArcticDB reads, excluding lazy setup."""
    reads: list[ObservedRead] = []
    original = Library.read

    def observe(self: Library, *args: Any, **kwargs: Any) -> Any:
        # Copy the read arguments ArcticDB executes because ArcticDB may modify the QueryBuilder
        recorded = copy.deepcopy(kwargs)
        # Perform the real ArcticDB read
        result = original(self, *args, **kwargs)

        # Ignore lazy setup calls; record only reads that return table data
        if not kwargs.get("lazy", False):
            assert isinstance(result.data, pa.Table)
            reads.append(ObservedRead(recorded, result.data))
        return result

    # Send read calls through observe() defined above, which records them and calls the real read
    monkeypatch.setattr(Library, "read", observe)
    return reads


@pytest.fixture
def batch_hint(monkeypatch: pytest.MonkeyPatch) -> Callable[[int | None], list[pl.DataFrame]]:
    """Let a test choose the reader's batch-size hint and inspect the DataFrames it produces."""
    original = pl.io.plugins.register_io_source

    def install(size: int | None) -> list[pl.DataFrame]:
        batches: list[pl.DataFrame] = []

        def register(io_source: Callable[..., Iterator[pl.DataFrame]], **kwargs: Any) -> pl.LazyFrame:
            def source(columns: Any, predicate: Any, n_rows: Any, supplied_size: Any) -> Iterator[pl.DataFrame]:
                # Override only the batch-size hint; keep Polars projection, predicate and row limit unchanged
                for batch in io_source(columns, predicate, n_rows, size):
                    batches.append(batch)
                    yield batch

            # Register the wrapped reader through the real Polars function
            return original(io_source=source, **kwargs)

        # Wrap registration before the test creates its scan
        monkeypatch.setattr(pl.io.plugins, "register_io_source", register)
        return batches

    return install


@pytest.fixture
def oracle():
    """Calculate the expected result using only Polars on known input data."""
    def expected_result(data: pl.DataFrame, operation: Callable[[pl.LazyFrame], pl.LazyFrame]) -> pl.DataFrame:
        # The reference never reads through Polarctic or uses its translator
        return operation(data.lazy()).collect(engine="in-memory")

    return expected_result