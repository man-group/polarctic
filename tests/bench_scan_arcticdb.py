"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source License, use of
this software will be governed by the Apache License, version 2.0.
"""

"""Compare Polarctic pushdown with a full ArcticDB read followed by Polars filtering."""

import datetime as dt
from contextlib import nullcontext

import polars as pl
import pytest
from arcticdb import Arctic, LibraryOptions, OutputFormat
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import collect_storage_stats, enable_polars_writes


@pytest.fixture
def benchmark_data(tmp_path):
    """Store one fixed version with several timeseries segments."""
    # Each benchmark case gets its own temporary local store
    uri = f"lmdb://{tmp_path / 'benchmark_store'}"
    ac = Arctic(uri, output_format=OutputFormat.POLARS)
    name = "benchmark"
    symbol = "data"
    lib = ac.create_library(
        name,
        library_options=LibraryOptions(rows_per_segment=10_000),
    )
    enable_polars_writes(lib)
    start = dt.datetime(2020, 1, 1)
    data = pl.DataFrame({
        "timestamp": pl.datetime_range(
            start,
            start + dt.timedelta(seconds=99_999),
            interval="1s",
            time_unit="ns",
            eager=True,
        ),
        "a": pl.Series(range(100_000), dtype=pl.Int64),
        "b": pl.Series(range(100_000), dtype=pl.Float64),
    })
    try:
        # Use the first column as its datetime index
        lib.write(symbol, data, index_column=True)
        yield lib, data, symbol
    finally:
        ac.delete_library(name)



# Compare a numeric filter and a filter on the stored datetime index
@pytest.mark.parametrize("query_name", ["value-filter", "time-range"])
@pytest.mark.parametrize("implementation", ["polarctic", "read-then-polars"])
@pytest.mark.parametrize("stats_enabled", [False, True], ids=["stats-off", "stats-on"])
def bench_scan_comparison(
    benchmark, benchmark_data, query_name, implementation, stats_enabled
):
    """Time the two workflows, with and without query-statistics recording."""
    # Get the local library, input data and symbol
    lib, data, symbol = benchmark_data

    if query_name == "value-filter":
        # Keep the final 10,000 rows using their numeric values
        predicate = pl.col("a") >= 90_000
    else:
        # Select the same rows by timestamp, 90,000 seconds after the start
        predicate = pl.col("timestamp") >= dt.datetime(2020, 1, 2, 1)

    # Apply the same filter and column selection in both workflows
    def apply(frame):
        return frame.filter(predicate).select("timestamp", "a")

    expected = apply(data.lazy()).collect(engine="in-memory")

    # Prepare a reusable Polarctic plan for the fixed stored version before timing starts.
    # These measurements cover warm/repeated collection, not cold storage
    query = apply(pc.scan_arcticdb(lib, symbol, as_of=0))

    def pushed():
        # Collect through Polarctic: the connector can send supported filtering
        # and column selection to ArcticDB before the backend returns the data
        return query.collect(engine="in-memory")

    def baseline():
        # Read the complete stored version from ArcticDB without backend filtering
        # or column selection, then apply both operations in Polars
        frame = lib.read(symbol, as_of=0, output_format=OutputFormat.POLARS).data
        return apply(frame.lazy()).collect(engine="in-memory")

    # Verify both workflows before measuring either of them
    assert_frame_equal(pushed(), expected)
    assert_frame_equal(baseline(), expected)
    run = pushed if implementation == "polarctic" else baseline

    if stats_enabled:
        # Record one separate, untimed execution for inspection.
        # Preserve the plain ArcticDB snapshot as benchmark metadata.
        with collect_storage_stats() as sample_stats:
            sample = run()
        assert_frame_equal(sample, expected)
        benchmark.extra_info["sample_query_stats"] = sample_stats

    # Enable recording once around the entire timing run.
    # Collector setup and snapshot retrieval are outside the timed callable.
    # The timed work includes recording overhead when enabled
    scope = collect_storage_stats() if stats_enabled else nullcontext()
    with scope:
        benchmark(run)

    benchmark.extra_info["dataset_rows"] = data.height
    benchmark.extra_info["arcticdb_version_requested"] = 0
    benchmark.extra_info["measurement"] = "warm repeated execution, in-memory engine"
    benchmark.extra_info["storage"] = "LMDB"

    # LMDB reads may not appear in query statistics.
    # Use the memory-backed unit tests to check data-segment read counts