"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

import sys
import gc
import datetime as dt
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from arcticdb import Arctic, LibraryOptions, OutputFormat
from arcticdb.version_store.library import Library

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
class ArcticSource:
    """Database details only; each test chooses its own data."""

    uri: str
    library_name: str
    library: Library


@pytest.fixture
def library_factory(tmp_path: Path):
    """Create isolated libraries with test-specific storage options."""
    created = []

    def create(
        *,
        storage: str = "lmdb",
        library_options: LibraryOptions | None = None,
    ) -> ArcticSource:
        """Create an empty library and return its connection details."""
        number = len(created)
        library_name = f"test_library_{number}"

        if storage == "lmdb":
            # Each connection gets a separate local directory
            uri = f"lmdb://{tmp_path / f'store_{number}'}"
        elif storage == "memory":
            # Each Arctic instance gets an isolated memory store
            uri = "mem://"
        else:
            raise ValueError(f"Unsupported test storage: {storage}")

        ac = Arctic(uri, output_format=OutputFormat.POLARS)
        lib = ac.create_library(
            library_name,
            library_options=library_options or LibraryOptions(),
        )

        created.append((ac, library_name))
        return ArcticSource(uri, library_name, lib)

    try:
        yield create
    finally:
        # Delete only libraries created by this fixture
        for ac, library_name in reversed(created):
            ac.delete_library(library_name)


@pytest.fixture
def arctic_source(library_factory) -> ArcticSource:
    """Create an empty temporary ArcticDB library with small segments for connector tests."""
    return library_factory(
        library_options=LibraryOptions(
            rows_per_segment=3,
            columns_per_segment=2,
            dynamic_schema=True,
        ),
    )


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Reusable input data, independent of any database."""
    return pl.DataFrame(
        {
            # Keep ts first so a test can choose index_column=True when writing
            "ts": pl.datetime_range(
                dt.datetime(2020, 1, 1),
                dt.datetime(2020, 1, 8),
                interval="1d",
                time_unit="ns",
                eager=True,
            ),
            "a": [0, 1, 2, 3, 4, 5, -5, -2],
            "b": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 8.0, 9.0],
            "desk": [
                "rates", "fx", "rates", "fx",
                "credit", "credit", "rates", "fx",
            ],
            "flag": [True, False, True, False, True, False, True, False],
            "text": [
                "alpha", 'a"b', "[x]", "plain",
                "beta", "x", "negative", "minus",
            ],
        },
        schema_overrides={"a": pl.Int64, "b": pl.Float64},
    )


@pytest.fixture(scope="session", autouse=True)
def windows_timezone_database():
    """Provide the timezone files required by native code on Windows."""
    if sys.platform == "win32":
        from pyarrow.util import download_tzdata_on_windows

        # Windows CI still requires this setup (even with ArcticDB 7.0.0).
        # Download them once before the tests run.
        download_tzdata_on_windows()