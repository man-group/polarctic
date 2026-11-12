"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source License, use of
this software will be governed by the Apache License, version 2.0.
"""

"""Compare ArcticDB filtering/column selection through Polarctic with full ArcticDB S3 reads followed by filtering/column selection in Polars."""

import datetime as dt
import uuid
from contextlib import nullcontext

import polars as pl
import pytest
from ahl.mongo import NativeMongoose
from arcticdb import OutputFormat
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import collect_storage_stats, storage_read_count


@pytest.fixture
def benchmark_data():
    """Create and clean up one temporary symbol in the managed test library."""
    # Open the managed library using my skibidi write permissions
    nm = NativeMongoose("research", api="v2")
    lib = nm["xodintsova.polarctic"]

    # Use a unique symbol so separate benchmark cases do not overwrite each other
    symbol = "benchmark_" + uuid.uuid4().hex
    written = False

    try:
        # Create one row per second, starting at midnight on January 1
        start = dt.datetime(2020, 1, 1)

        # timestamp is first so index_column=True makes it the timeseries index;
        # a is used for filtering; b is omitted from the query output
        data = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    start,
                    start + dt.timedelta(seconds=99_999),
                    interval="1s",
                    time_unit="ns",
                    eager=True,
                ),
                "a": pl.Series(range(100_000), dtype=pl.Int64),
                "b": pl.Series(range(100_000), dtype=pl.Float64),
            }
        )

        # Write the table directly from Polars, creating version 0
        lib.write(symbol, data, index_column=True)
        written = True

        yield lib, data, symbol

    finally:
        # Clean up even if validation or benchmarking fails (delete only the temporary symbol)
        if written:
            lib.delete(symbol)


# Compare a numeric filter and a filter on the stored datetime index
@pytest.mark.parametrize("query_name", ["value-filter", "time-range"])
@pytest.mark.parametrize("implementation", ["polarctic", "read-then-polars"])
@pytest.mark.parametrize("stats_enabled", [False, True], ids=["stats-off", "stats-on"])
def bench_scan_comparison(
    benchmark, benchmark_data, query_name, implementation, stats_enabled
):
    """Measure equivalent queries through Polarctic and direct-read workflows."""
    # Get the managed library, original input table and unique symbol name
    lib, data, symbol = benchmark_data

    if query_name == "value-filter":
        # Keep the final 10,000 rows using their numeric values
        predicate = pl.col("a") >= 90_000
    else:
        # Select the same rows by timestamp, 90,000 seconds after the start
        predicate = pl.col("timestamp") >= dt.datetime(2020, 1, 2, 1)

    def apply(frame):
        # Apply the same filter and column selection in both workflows
        return frame.filter(predicate).select("timestamp", "a")

    expected = apply(data.lazy()).collect(engine="in-memory")

    # Prepare a Polarctic query for the fixed stored version before timing starts.
    # Repeated collections below reuse this prepared plan
    query = apply(pc.scan_arcticdb(lib, symbol, as_of=0))

    def pushed():
        # Collect through Polarctic: the connector can send supported filtering
        # and column selection to ArcticDB before the backend returns the data
        return query.collect(engine="in-memory")

    def baseline():
        # Read the complete stored version from ArcticDB without backend filtering
        # or column selection, then apply both operations in Polars
        frame = lib.read(
            symbol,
            as_of=0,
            output_format=OutputFormat.POLARS,
        ).data
        return apply(frame.lazy()).collect(engine="in-memory")

    # Verify both workflows before measuring either of them
    assert_frame_equal(pushed(), expected)
    assert_frame_equal(baseline(), expected)

    run = pushed if implementation == "polarctic" else baseline

    if stats_enabled:
        # Confirm that the managed storage reports S3 table-data reads
        with collect_storage_stats() as control:
            lib.read(
                symbol,
                as_of=0,
                output_format=OutputFormat.POLARS,
            )

        if storage_read_count(control, "TABLE_DATA") <= 0:
            raise RuntimeError(
                "S3 data reads were not recorded by query statistics"
            )

        # Record one execution separately to inspect its storage statistics
        # outside the timed measurement
        with collect_storage_stats() as sample_stats:
            sample = run()

        assert_frame_equal(sample, expected)
        benchmark.extra_info["sample_query_stats"] = sample_stats

    # For stats-on cases, record operations during the timed work.
    # Collector setup and snapshot retrieval happen outside the timed callable.
    # For stats-off cases, nullcontext does nothing
    scope = collect_storage_stats() if stats_enabled else nullcontext()
    with scope:
        # pytest-benchmark repeatedly calls the chosen workflow and measures
        # how long each execution takes
        benchmark(run)

    benchmark.extra_info["dataset_rows"] = data.height
    benchmark.extra_info["arcticdb_version_requested"] = 0
    benchmark.extra_info["query"] = query_name
    benchmark.extra_info["implementation"] = implementation
    benchmark.extra_info["query_stats_enabled"] = stats_enabled
    benchmark.extra_info["storage"] = "managed S3-compatible library"
    benchmark.extra_info["library"] = "xodintsova.polarctic"

    # The earlier validation reads may have warmed caches, so these results
    # measure repeated execution, not cold-storage reads
    benchmark.extra_info["measurement"] = ("warm repeated execution, in-memory engine")