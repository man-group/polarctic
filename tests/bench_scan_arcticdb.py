"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source License, use of
this software will be governed by the Apache License, version 2.0.
"""

"""Compare Polarctic pushdown with a full ArcticDB read followed by Polars filtering."""

import datetime as dt

import polars as pl
import pytest
from arcticdb import LibraryOptions, OutputFormat
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import override_batch_hint


@pytest.fixture
def benchmark_data(library_factory):
    """Create a local benchmark source containing 100,000 timestamped rows."""
    # Use the shared factory, but choose larger segments for this dataset
    source = library_factory(library_options=LibraryOptions(rows_per_segment=10_000))

    start = dt.datetime(2020, 1, 1)
    data = pl.DataFrame(
        {
            # One row per second, with timestamp first for index_column=True
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

    source.library.write("data", data, index_column=True)
    return source, data


# Compare a numeric filter and a filter on the stored datetime index
@pytest.mark.parametrize("query_name", ["value-filter", "time-range"])
@pytest.mark.parametrize("implementation", ["polarctic", "read-then-polars"])
@pytest.mark.parametrize(
    "batch_hint",
    [
        pytest.param(None, id="single-read"),
        pytest.param(5_000, id="hint-5000"),
        pytest.param(10_000, id="hint-10000"),
        pytest.param(25_000, id="hint-25000"),
    ],
)
def bench_scan_comparison(benchmark, benchmark_data, query_name, implementation, batch_hint):
    """Measure repeated collection with different Polarctic batch hints."""
    source, data = benchmark_data
    lib = source.library

    if query_name == "value-filter":
        # Keep the final 10,000 rows using their numeric values
        predicate = pl.col("a") >= 90_000
    else:
        # Request the same final 10,000 rows using their timestamps (90,000 seconds after the start)
        predicate = pl.col("timestamp") >= dt.datetime(2020, 1, 2, 1)

    # Apply the same filter and column selection in both workflows
    def apply(frame):
        return frame.filter(predicate).select("timestamp", "a")

    # Calculate the expected answer without using the connector
    expected = apply(data.lazy()).collect(engine="streaming")

    # Prepare the scan before timing starts
    with override_batch_hint(batch_hint, record_batches=False):
        query = apply(pc.scan_arcticdb(lib, "data"))

    def pushed():
        """Collect through Polarctic using the prepared scan."""
        return query.collect(engine="streaming")

    def baseline():
        """Read the full table, then filter and select using Polars."""
        frame = lib.read("data", output_format=OutputFormat.POLARS).data
        return apply(frame.lazy()).collect(engine="streaming")

    # Verify both workflows before measuring either of them
    assert_frame_equal(pushed(), expected)
    assert_frame_equal(baseline(), expected)

    run = pushed if implementation == "polarctic" else baseline

    # The plan and fixture setup are outside the timed function.
    # This measures warm, repeated execution—not cold storage access
    benchmark(run)


    # These fields are included in pytest-benchmark's JSON output
    benchmark.extra_info["dataset_rows"] = data.height
    benchmark.extra_info["batch_hint"] = batch_hint
    benchmark.extra_info["engine"] = "streaming"
    benchmark.extra_info["storage"] = "LMDB"
    benchmark.extra_info["measurement"] = "warm repeated execution"

    # The hint controls Polarctic's reader, not the full-read baseline.
    # The baseline is repeated for each hint to keep the cases paired
    benchmark.extra_info["hint_applies_to"] = "Polarctic reader only"