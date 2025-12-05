import pytest
import shutil
from polarctic.polarctic import PolarsToArcticDBTranslator
from arcticdb import Arctic
from pathlib import Path
import pandas as pd
import numpy as np

@pytest.fixture
def translator():
    return PolarsToArcticDBTranslator()

@pytest.fixture(scope="function")
def init_arcticdb(tmp_path):
    """
    Initialize a real ArcticDB LMDB-backed store and write two pandas DataFrames
    into a library.

    Yields a dict with:
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
    df1 = pd.DataFrame({"a": np.arange(10, dtype=np.int64), "b": np.arange(10, 20, dtype=np.float64), "ts": pd.date_range("2020-01-01", periods=10)})
    df2 = pd.DataFrame({"a": np.arange(2, 12, dtype=np.float64), "b": np.arange(15, 25, dtype=np.int64), "ts": pd.date_range("2020-02-01", periods=10)})

    # Write DataFrames into the library under symbols "df1" and "df2".
    lib.write("df1", df1)
    lib.write("df2", df2)

    yield {
        "uri": uri,
        "lmdb_path": lmdb_path,
        "lib_name": lib_name,
        "tables": {"df1": df1, "df2": df2},
        "ac": ac,
        "lib": lib,
    }

@pytest.fixture(scope="function")
def delete_arcticdb(init_arcticdb):
    """
    Teardown fixture: remove the LMDB directory created by init_arcticdb.
    Tests should include this fixture in the parameter list to ensure cleanup.
    """
    yield
    # delete library
    ac = init_arcticdb["ac"]
    lib_name = init_arcticdb["lib_name"]
    ac.delete_library(lib_name)

    # remove directory
    lmdb_path = init_arcticdb["lmdb_path"]
    try:
        shutil.rmtree(lmdb_path)
    except Exception:
        # best-effort cleanup; ignore errors
        pass
