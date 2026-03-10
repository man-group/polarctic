"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

import gc
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from arcticdb import Arctic

from polarctic.polarctic import PolarsToArcticDBTranslator


@pytest.fixture
def translator() -> PolarsToArcticDBTranslator:
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
