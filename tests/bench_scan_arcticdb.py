"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.

Execution-focused benchmarks for polarctic.scan_arcticdb() using a real
LMDB-backed ArcticDB store.

Construction-only benchmarks live in tests/bench_scan_arcticdb_construction.py.

Run with:
    pytest tests/bench_scan_arcticdb.py -v --benchmark-only
"""

import gc
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
from arcticdb import Arctic, OutputFormat, QueryBuilder
from pytest_benchmark.fixture import BenchmarkFixture

import polarctic.polarctic as polarctic_module

_SIMPLE_FILTER_EXPR = pl.col("a") > 500
_COMPOUND_FILTER_EXPR = (pl.col("a") > 200) & (pl.col("b") < 700.0)
_PREFILTER_EXTRA_EXPR = pl.col("b") < 700.0
_TWO_COLUMN_PROJECTION = ("a", "b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_simple_filter_qb() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[qb["a"] > 500]
    return cast(QueryBuilder, qb)


def _build_compound_filter_qb() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[(qb["a"] > 200) & (qb["b"] < 700.0)]
    return cast(QueryBuilder, qb)


def _build_prefilter_qb() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[qb["a"] > 200]
    return cast(QueryBuilder, qb)


# ---------------------------------------------------------------------------
# Fixtures (module-scoped to pay setup cost once per session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def arcticdb_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """
    Create a real ArcticDB LMDB store populated with DataFrames of varying
    sizes so individual benchmarks can pick the most representative one.
    """
    lmdb_dir: Path = tmp_path_factory.mktemp("arctic_bench") / "lmdb"
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    uri = f"lmdb://{lmdb_dir}"
    lib_name = "bench_lib"

    ac = Arctic(uri)
    lib = ac.create_library(lib_name)

    rng = np.random.default_rng(42)

    def _make_df(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": rng.integers(0, 1000, size=n).astype(np.int64),
                "b": rng.uniform(0.0, 1000.0, size=n).astype(np.float64),
                "label": [f"item_{i % 50}" for i in range(n)],
                "ts": pd.date_range("2020-01-01", periods=n, freq="s"),
            }
        )

    small_symbol = "small"
    medium_symbol = "medium"
    large_symbol = "large"

    lib.write(small_symbol, _make_df(100))
    lib.write(medium_symbol, _make_df(10_000))
    lib.write(large_symbol, _make_df(100_000))

    yield {
        "uri": uri,
        "lib_name": lib_name,
        "lib": lib,
        "ac": ac,
        "small_symbol": small_symbol,
        "medium_symbol": medium_symbol,
        "large_symbol": large_symbol,
        "lmdb_dir": str(lmdb_dir),
    }

    # Teardown
    lib_ref = lib
    del lib_ref
    gc.collect()


@pytest.fixture(scope="module")
def prepared_query_builders() -> dict[str, QueryBuilder]:
    """Create QBs once so benchmark iterations measure scan/read execution only."""
    return {
        "simple": _build_simple_filter_qb(),
        "compound": _build_compound_filter_qb(),
        "prefilter": _build_prefilter_qb(),
    }


@pytest.fixture(scope="module")
def prepared_lazy_frames(
    arcticdb_store: dict[str, Any],
    prepared_query_builders: dict[str, QueryBuilder],
) -> dict[str, pl.LazyFrame]:
    """Prepare LazyFrame plans once so per-iteration timing focuses on collect()."""
    lib = arcticdb_store["lib"]
    small_symbol = arcticdb_store["small_symbol"]
    medium_symbol = arcticdb_store["medium_symbol"]
    large_symbol = arcticdb_store["large_symbol"]

    lazy_small = polarctic_module.scan_arcticdb(lib, small_symbol)
    lazy_medium = polarctic_module.scan_arcticdb(lib, medium_symbol)
    lazy_large = polarctic_module.scan_arcticdb(lib, large_symbol)

    arctic_lazy_medium = lib.read(medium_symbol, lazy=True, output_format=OutputFormat.PYARROW)
    arctic_lazy_prefilter = lib.read(
        medium_symbol,
        query_builder=prepared_query_builders["prefilter"],
        lazy=True,
        output_format=OutputFormat.PYARROW,
    )

    return {
        "full_small": lazy_small,
        "full_medium": lazy_medium,
        "full_large": lazy_large,
        "filter_simple_medium": lazy_medium.filter(_SIMPLE_FILTER_EXPR),
        "filter_compound_medium": lazy_medium.filter(_COMPOUND_FILTER_EXPR),
        "filter_simple_large": lazy_large.filter(_SIMPLE_FILTER_EXPR),
        "filter_compound_large": lazy_large.filter(_COMPOUND_FILTER_EXPR),
        "select_two_columns_medium": lazy_medium.select(*_TWO_COLUMN_PROJECTION),
        "select_two_columns_large": lazy_large.select(*_TWO_COLUMN_PROJECTION),
        "lazy_source_medium": polarctic_module.scan_arcticdb(arctic_lazy_medium),
        "lazy_prefilter_medium": polarctic_module.scan_arcticdb(arctic_lazy_prefilter).filter(
            _PREFILTER_EXTRA_EXPR
        ),
    }


# ---------------------------------------------------------------------------
# Full scan (no filter, no projection)
# ---------------------------------------------------------------------------


def bench_full_scan_small(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    benchmark(prepared_lazy_frames["full_small"].collect)


def bench_full_scan_medium(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    benchmark(prepared_lazy_frames["full_medium"].collect)


def bench_full_scan_large(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    benchmark(prepared_lazy_frames["full_large"].collect)


# ---------------------------------------------------------------------------
# Filter pushdown
# ---------------------------------------------------------------------------


def bench_filter_simple_medium(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    """Single comparison filter pushed down to ArcticDB."""
    benchmark(prepared_lazy_frames["filter_simple_medium"].collect)


def bench_filter_compound_medium(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    """Compound AND filter pushed down to ArcticDB."""
    benchmark(prepared_lazy_frames["filter_compound_medium"].collect)


def bench_filter_simple_large(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    benchmark(prepared_lazy_frames["filter_simple_large"].collect)


def bench_filter_compound_large(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    benchmark(prepared_lazy_frames["filter_compound_large"].collect)


# ---------------------------------------------------------------------------
# Column projection (select pushdown)
# ---------------------------------------------------------------------------


def bench_select_two_columns_medium(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    benchmark(prepared_lazy_frames["select_two_columns_medium"].collect)


def bench_select_two_columns_large(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    benchmark(prepared_lazy_frames["select_two_columns_large"].collect)


# ---------------------------------------------------------------------------
# LazyDataFrame source (ArcticDB lazy read handed to scan_arcticdb)
# ---------------------------------------------------------------------------


def bench_lazy_dataframe_source_medium(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    benchmark(prepared_lazy_frames["lazy_source_medium"].collect)


def bench_lazy_dataframe_with_prefilter_medium(
    benchmark: BenchmarkFixture,
    prepared_lazy_frames: dict[str, pl.LazyFrame],
) -> None:
    """ArcticDB QB pre-filter plus an extra pushed-down Polars predicate."""
    benchmark(prepared_lazy_frames["lazy_prefilter_medium"].collect)


# ---------------------------------------------------------------------------
# parse_schema
# ---------------------------------------------------------------------------


def bench_parse_schema(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
) -> None:
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["medium_symbol"]
    benchmark(lambda: polarctic_module.parse_schema(lib, symbol))


# ---------------------------------------------------------------------------
# ArcticDB baselines - direct lib.read() without polarctic
# ---------------------------------------------------------------------------


def bench_baseline_full_read_medium(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
) -> None:
    """Direct ArcticDB read, medium dataset - baseline for bench_full_scan_medium."""
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["medium_symbol"]
    benchmark(lambda: lib.read(symbol, output_format=OutputFormat.PANDAS).data)


def bench_baseline_full_read_large(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
) -> None:
    """Direct ArcticDB read, large dataset - baseline for bench_full_scan_large."""
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["large_symbol"]
    benchmark(lambda: lib.read(symbol, output_format=OutputFormat.PANDAS).data)


def bench_baseline_filter_simple_medium(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
    prepared_query_builders: dict[str, QueryBuilder],
) -> None:
    """Direct ArcticDB QB filter, medium dataset - baseline for bench_filter_simple_medium."""
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["medium_symbol"]
    benchmark(
        lambda: (
            lib.read(
                symbol,
                query_builder=prepared_query_builders["simple"],
                output_format=OutputFormat.PANDAS,
            ).data
        )
    )


def bench_baseline_filter_simple_large(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
    prepared_query_builders: dict[str, QueryBuilder],
) -> None:
    """Direct ArcticDB QB filter, large dataset - baseline for bench_filter_simple_large."""
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["large_symbol"]
    benchmark(
        lambda: (
            lib.read(
                symbol,
                query_builder=prepared_query_builders["simple"],
                output_format=OutputFormat.PANDAS,
            ).data
        )
    )


def bench_baseline_filter_compound_medium(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
    prepared_query_builders: dict[str, QueryBuilder],
) -> None:
    """Direct ArcticDB QB compound filter, medium dataset baseline."""
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["medium_symbol"]
    benchmark(
        lambda: (
            lib.read(
                symbol,
                query_builder=prepared_query_builders["compound"],
                output_format=OutputFormat.PANDAS,
            ).data
        )
    )


def bench_baseline_filter_compound_large(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
    prepared_query_builders: dict[str, QueryBuilder],
) -> None:
    """Direct ArcticDB QB compound filter, large dataset baseline."""
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["large_symbol"]
    benchmark(
        lambda: (
            lib.read(
                symbol,
                query_builder=prepared_query_builders["compound"],
                output_format=OutputFormat.PANDAS,
            ).data
        )
    )


def bench_baseline_select_two_columns_medium(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
) -> None:
    """Direct ArcticDB projection, medium dataset baseline."""
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["medium_symbol"]
    benchmark(
        lambda: (
            lib.read(
                symbol,
                columns=list(_TWO_COLUMN_PROJECTION),
                output_format=OutputFormat.PANDAS,
            ).data
        )
    )


def bench_baseline_select_two_columns_large(
    benchmark: BenchmarkFixture,
    arcticdb_store: dict[str, Any],
) -> None:
    """Direct ArcticDB projection, large dataset baseline."""
    lib = arcticdb_store["lib"]
    symbol = arcticdb_store["large_symbol"]
    benchmark(
        lambda: (
            lib.read(
                symbol,
                columns=list(_TWO_COLUMN_PROJECTION),
                output_format=OutputFormat.PANDAS,
            ).data
        )
    )
