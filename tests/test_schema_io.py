"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Tests 8: verify planning, index reuse and column-selection IO using S3 query statistics."""

import datetime as dt
import os
import uuid

import pandas as pd
import polars as pl
import pytest
from arcticdb import Arctic, LibraryOptions, OutputFormat
from arcticdb_ext.storage import KeyType
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import backend_frame, collect_storage_stats, observe_reads, oracle, override_batch_hint, storage_read_count


@pytest.fixture
def stats_library():
    """Create a temporary library on an approved S3 test store."""
    # Use a dedicated disposable test store
    uri = os.environ.get("POLARCTIC_TEST_STATS_URI")
    if not uri:
        pytest.skip("Set POLARCTIC_TEST_STATS_URI to an approved disposable S3 test store")
    if not uri.lower().startswith("s3"):
        pytest.fail("These tests require S3 GetObject counters, not LMDB")

    # Request Polars output from libraries opened through this connection
    ac = Arctic(uri, output_format=OutputFormat.POLARS)

    # Use a unique name so separate test runs do not share a library
    name = "polarctic_stats_" + uuid.uuid4().hex

    # Use small row and column segments so this small dataset spans several stored segments
    lib = ac.create_library(name, library_options=LibraryOptions(rows_per_segment=3, columns_per_segment=2))
    try:
        # Give the empty library to the test (the test then decides which data to write)
        yield lib
    finally:
        # delete only the library created by this fixture
        ac.delete_library(name)


def assert_data_reads_are_recorded(lib):
    """Check that the trackers actually record a real table-data read."""
    # Perform a real full read while statistics collection is enabled.
    # The calling test has already written symbol data as version 0
    with collect_storage_stats() as stats:
        lib.read("data", as_of=0, output_format=OutputFormat.POLARS)

    # Check that at least one TABLE_DATA fetch must have been recorded
    assert storage_read_count(stats, "TABLE_DATA") > 0, (
        "The tracker recorded no TABLE_DATA reads; zero counts cannot establish IO behaviour"
    )


def data_segment_layout(lib, symbol, version):
    """Get metadata describing the stored data segments."""
    # ArcticDB 7.0 exposes the toolbox through the Library object.
    # The toolbox returns pandas metadata, not query-result data
    tool = lib.library_tool()
    layout = tool.read_index(symbol, as_of=version)

    # Keep only entries describing table-data segments
    layout = layout[layout["key_type"] == int(KeyType.TABLE_DATA)]

    if layout.empty:
        raise RuntimeError("No stored data-segment layout was found")

    return layout



def test_schema_discovery_reads_no_table_data_keys(stats_library, sample_data):
    """Check that planning fetches no table-data segments from S3."""
    # Store the input before measuring planning
    lib = stats_library
    lib.write("data", sample_data)

    # First check that real table-data reads produce counters
    assert_data_reads_are_recorded(lib)

    # Record from scan creation onwards (this includes any schema lookup triggered while adding the operations)
    with collect_storage_stats() as planning:
        query = pc.scan_arcticdb(lib, "data", as_of=0).filter(pl.col("a") > 1).select("a")
        assert query.collect_schema() == pl.Schema({"a": pl.Int64})

    # Planning should not fetch the stored table-data segments
    assert storage_read_count(planning, "TABLE_DATA") == 0

    # Fetching the index during planning is expected in this setup.
    # A positive count also confirms that this phase recorded index IO
    assert storage_read_count(planning, "TABLE_INDEX") > 0


@pytest.mark.parametrize("hint", [None, 3], ids=["single-read", "batched"])
def test_collection_reuses_schema_index(stats_library, sample_data, hint):
    """Check index reuse during single-read and batched collection."""
    lib = stats_library
    lib.write("data", sample_data)
    assert_data_reads_are_recorded(lib)

    # Create the scan with the selected hint, and record the batches delivered by the reader
    with override_batch_hint(hint) as batches:

        # Measure planning separately from collection
        with collect_storage_stats() as planning:
            query = pc.scan_arcticdb(lib, "data", as_of=0)
            assert query.collect_schema() == sample_data.schema

        # Start fresh counters for collecting the same query
        with collect_storage_stats() as collection:
            actual = query.collect(engine="streaming")

    # Check that reading preserved all values, columns and types
    assert_frame_equal(actual, sample_data)

    if hint is not None:
        # Verify that the batched case delivered multiple output batches.
        # This does not count underlying storage operations
        assert len(batches) >= 2

    # Planning should have loaded the index in this configured setup
    assert storage_read_count(planning, "TABLE_INDEX") > 0

    # Collection should have read actual table data
    assert storage_read_count(collection, "TABLE_DATA") > 0

    # The index obtained during planning should be reused during collection
    assert storage_read_count(collection, "TABLE_INDEX") == 0


@pytest.mark.parametrize("hint", [1, 2, 3, 4])
def test_small_hints_do_not_repeat_table_data_fetches(stats_library, hint):
    """Check that small batch hints do not cause unnecessary data fetches."""
    lib = stats_library

    # Use only one column so column selection does not affect this test.
    # Write and append separately to create different stored row-slice sizes
    first = pl.DataFrame({"a": list(range(5))})
    later = pl.DataFrame({"a": list(range(5, 10))})

    lib.write("data", first)   # Create version 0
    lib.append("data", later)   # Create version 1

    # Version 1 contains all ten rows, in the original order
    expected = pl.concat([first, later])

    # Read metadata describing the stored data segments before measuring the query 
    # so it is not counted as part of the query's collection
    layout = data_segment_layout(lib, "data", 1)

    # Check that at least one segment covers three or more rows (so hints of 1 and 2 are smaller than that slice)
    if (layout["end_row"] - layout["start_row"]).max() < 3:
        raise RuntimeError("The setup did not produce the intended row slices")

    stored_segments = len(layout)

    # First measure a direct full-table read to checks how the counters behave for the known stored layout
    with collect_storage_stats() as baseline_stats:
        baseline = lib.read(
            "data",
            as_of=1,
            output_format=OutputFormat.POLARS,
        ).data

    assert_frame_equal(baseline, expected)

    # Get the number of recorded table-data fetches
    baseline_count = storage_read_count(baseline_stats, "TABLE_DATA")

    # A full read is expected to have one fetch per required data segment.
    # Check that the counters actually behave that way
    if baseline_count != stored_segments:
        raise RuntimeError(
            "Cannot calibrate data-fetch counts: "
            f"expected {stored_segments}, recorded {baseline_count}. "
            "Check backend caching and query-statistics behaviour."
        )

    # Now read the same version through Polarctic
    with override_batch_hint(hint):
        query = pc.scan_arcticdb(lib, "data", as_of=1)

        # Discover column names and types before measuring collection
        query.collect_schema()

        # Record only the storage work performed during collection
        with collect_storage_stats() as scan_stats:
            actual = query.collect(engine="streaming")

    # Batching must preserve every row, its order and its type
    assert_frame_equal(actual, expected)

    # Get the data-fetch count for the Polarctic scan
    scan_count = storage_read_count(scan_stats, "TABLE_DATA")

    # The complete scan needs the same stored segments as the reference read.
    # Its small output hint should not cause extra segment fetches
    assert scan_count == stored_segments, (
        f"Expected {stored_segments} data fetches, recorded {scan_count}"
    )


def test_projection_reduces_table_data_reads(stats_library, sample_data):
    """Check that a narrow projection fetches fewer data segments than a full read."""
    lib = stats_library
    lib.write("data", sample_data)
    assert_data_reads_are_recorded(lib)

    # The fixture stores six columns in two-column segments.
    # Hold the version and batch hint fixed; change only the requested columns
    with override_batch_hint(None):
        full_query = pc.scan_arcticdb(lib, "data", as_of=0)
        narrow_query = pc.scan_arcticdb(lib, "data", as_of=0).select("a")

        # Resolve schemas before measuring collection
        full_query.collect_schema()
        narrow_query.collect_schema()

        # Measure the full-table collection
        with collect_storage_stats() as full_stats:
            full = full_query.collect()

        # Measure the one-column collection with fresh counters
        with collect_storage_stats() as narrow_stats:
            narrow = narrow_query.collect()

    # First verify that both queries return the correct data
    assert_frame_equal(full, sample_data)
    assert_frame_equal(narrow, sample_data.select("a"))

    # Read the number of recorded table-data fetches for each query
    full_count = storage_read_count(full_stats, "TABLE_DATA")
    narrow_count = storage_read_count(narrow_stats, "TABLE_DATA")

    # The narrow read must fetch some data, but fewer segments than the full read
    assert 0 < narrow_count < full_count


def test_timeseries_index_filter_returns_correct_rows_and_skips_slices(stats_library):
    """Check correct time filtering and avoidance of out-of-range data fetches."""
    lib = stats_library

    # # Create one row per day, from January 1 through January 12.
    # Put timestamp first; index_column=True uses the first column as the index
    data = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                dt.datetime(2020, 1, 1),
                dt.datetime(2020, 1, 12),
                interval="1d",
                time_unit="ns",
                eager=True,
            ),
            "a": list(range(12)),
        }
    )

    lib.write("data", data, index_column=True)

    # Read metadata describing the stored data segments before measuring the query 
    # so it is not counted as part of the query's collection
    layout = data_segment_layout(lib, "data", 0)

    # Choose a range covering January 4, 5 and 6
    start = dt.datetime(2020, 1, 4)
    end = dt.datetime(2020, 1, 7)

    predicate = (
        (pl.col("timestamp") >= start)
        & (pl.col("timestamp") < end)
    )
    expected = oracle(data, lambda frame: frame.filter(predicate))

    # Calculate which stored segments overlap the requested range
    # (the ones that start before our requested end, or end after our requested start).
    # The toolbox exposes these boundaries as pandas timestamps
    overlaps = (
        (layout.index < pd.Timestamp(end))
        & (layout["end_index"] > pd.Timestamp(start))
    )
    required_segments = int(overlaps.sum())

    # Make sure the setup creates the correct testing conditions:
    # some stored segments are needed, and some should be skipped
    if not 0 < required_segments < len(layout):
        raise RuntimeError("The test interval must exclude whole stored slices")

    # First measure a direct full-table read to checks how the counters behave for the known stored layout
    with collect_storage_stats() as full_stats:
        full = lib.read(
            "data",
            as_of=0,
            output_format=OutputFormat.POLARS,
        ).data

    assert_frame_equal(full, data)

    # One fetch per data segment is expected
    assert storage_read_count(full_stats, "TABLE_DATA") == len(layout)

    # Disable batching to focus on time filtering and storage skipping
    with override_batch_hint(None):
        query = pc.scan_arcticdb(lib, "data", as_of=0).filter(predicate)

        # Discover column names and types before measuring collection
        query.collect_schema()

        # Record the read requests and tables returned by ArcticDB
        with observe_reads() as reads:
            # Also record the S3 operations made during collection
            with collect_storage_stats() as filtered_stats:
                actual = query.collect()

    # ArcticDB must already have returned the correctly filtered rows
    assert_frame_equal(backend_frame(reads), expected)

    # The final Polarctic result must also be correct
    assert_frame_equal(actual, expected)

    # Check storage work separately: only the overlapping data segments should need to be fetched
    assert storage_read_count(
        filtered_stats, "TABLE_DATA"
    ) == required_segments


def test_stats_enabled_does_not_change_results(stats_library, sample_data):
    """Check that collecting statistics does not change the query result."""
    lib = stats_library
    lib.write("data", sample_data)
    assert_data_reads_are_recorded(lib)

    def apply(frame):
        # Use the same filter and column order for both executions
        return frame.filter(pl.col("a") > 1).select("b", "a")

    # Disable batching to keep the comparison focused on statistics collection
    with override_batch_hint(None):
        # Run a fresh query without enabling statistics
        without_stats = apply(pc.scan_arcticdb(lib, "data", as_of=0)).collect()

        # Run the equivalent query with statistics enabled (both queries read the same stored version)
        with collect_storage_stats() as stats:
            with_stats = apply(pc.scan_arcticdb(lib, "data", as_of=0)).collect()

    # Calculate the expected answer independently using only Polars
    expected = oracle(sample_data, apply)

    # Both executions must have the same correct values, order and types
    assert_frame_equal(without_stats, expected)
    assert_frame_equal(with_stats, expected)

    # Confirm that the measured execution actually recorded table-data reads
    assert storage_read_count(stats, "TABLE_DATA") > 0
