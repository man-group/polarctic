"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Check planning, index reuse and column reads using memory storage."""

import datetime as dt

import pandas as pd
import polars as pl
import pytest
from arcticdb import Arctic, LibraryOptions, OutputFormat, QueryBuilder
from arcticdb_ext.storage import KeyType
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import collect_storage_stats, enable_polars_writes, observe_reads, oracle, override_batch_hint, query_stats_operation_count


@pytest.fixture
def stats_library():
    """Create an empty, isolated in-memory library for each test."""
    # Public ArcticDB storage 
    ac = Arctic("mem://", output_format=OutputFormat.POLARS)

    # Each fixture has its own store, so fixed names are sufficient
    library_name = "stats_tests"
    symbol = "data"

    lib = ac.create_library(
        library_name,
        library_options=LibraryOptions(
            rows_per_segment=3,
            columns_per_segment=2,
        ),
    )

    enable_polars_writes(lib)

    try:
        # Each test writes the data needed for its own assertions
        yield lib, symbol
    finally:
        # Delete only the library created by this fixture
        ac.delete_library(library_name)


def assert_data_reads_are_recorded(lib, symbol, version=0):
    """Check that the trackers record a real read of this test's symbol."""
    # Read the specified version while statistics are enabled
    with collect_storage_stats() as stats:
        lib.read(
            symbol,
            as_of=version,
            output_format=OutputFormat.POLARS,
        )

    # Zero counts cannot prove anything if recording is not working
    if query_stats_operation_count(stats, "Memory_GetObject", "TABLE_DATA") <= 0:
        raise RuntimeError(
            "The tracker recorded no TABLE_DATA reads; "
            "check storage access and query-statistics support"
        )


def data_segment_layout(lib, symbol, version):
    """Get metadata describing the stored data segments."""
    # Access the toolbox through the underlying version store.
    # It returns stored-segment metadata, not query-result data
    tool = lib._nvs.library_tool()
    layout = tool.read_index(symbol, as_of=version)

    # Keep only entries describing table-data segments
    layout = layout[layout["key_type"] == int(KeyType.TABLE_DATA)]

    if layout.empty:
        raise RuntimeError("No stored data-segment layout was found")

    return layout


def test_schema_discovery_reads_no_table_data_keys(stats_library, sample_data):
    """Check that planning reads no table-data segments."""
    # Get this test's library and symbol
    lib, symbol = stats_library
    lib.write(symbol, sample_data)

    # First check that real table-data reads produce counters
    assert_data_reads_are_recorded(lib, symbol)

    # Record from scan creation onwards (this includes any schema lookup triggered while adding the operations)
    with collect_storage_stats() as planning:
        query = pc.scan_arcticdb(lib, symbol, as_of=0).filter(pl.col("a") > 1).select("a")
        assert query.collect_schema() == pl.Schema({"a": pl.Int64})

    # Planning should not fetch the stored table-data segments
    assert query_stats_operation_count(planning, "Memory_GetObject", "TABLE_DATA") == 0

    # Fetching the index during planning is expected in this setup.
    # A positive count also confirms that this phase recorded index IO
    assert query_stats_operation_count(planning, "Memory_GetObject", "TABLE_INDEX") > 0


@pytest.mark.parametrize("hint", [None, 3], ids=["single-read", "batched"])
def test_collection_reuses_schema_index(stats_library, sample_data, hint):
    """Check index reuse during single-read and batched collection."""
    # Get this test's library and symbol
    lib, symbol = stats_library
    lib.write(symbol, sample_data)
    assert_data_reads_are_recorded(lib, symbol)

    # Create the scan with the selected hint, and record the batches delivered by the reader
    with override_batch_hint(hint) as batches:

        # Measure planning separately from collection
        with collect_storage_stats() as planning:
            query = pc.scan_arcticdb(lib, symbol, as_of=0)
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
    assert query_stats_operation_count(planning, "Memory_GetObject", "TABLE_INDEX") > 0

    # Collection should have read actual table data
    assert query_stats_operation_count(collection, "Memory_GetObject", "TABLE_DATA") > 0

    # The index obtained during planning should be reused during collection
    assert query_stats_operation_count(collection, "Memory_GetObject", "TABLE_INDEX") == 0


@pytest.mark.parametrize("hint", [1, 2, 3, 4])
def test_small_hints_do_not_repeat_table_data_fetches(stats_library, hint):
    """Check that small batch hints do not cause unnecessary data fetches."""
    # Get this test's library and symbol
    lib, symbol = stats_library

    # Use only one column so column selection does not affect this test.
    # Write and append separately to create different stored row-slice sizes
    first = pl.DataFrame({"a": list(range(5))})
    later = pl.DataFrame({"a": list(range(5, 10))})

    lib.write(symbol, first)   # Create version 0
    lib.append(symbol, later)   # Create version 1

    # Version 1 contains all ten rows, in the original order
    expected = pl.concat([first, later])

    # Read metadata describing the stored data segments before measuring the query 
    # so it is not counted as part of the query's collection
    layout = data_segment_layout(lib, symbol, 1)

    # Check that at least one segment covers three or more rows (so hints of 1 and 2 are smaller than that slice)
    if (layout["end_row"] - layout["start_row"]).max() < 3:
        raise RuntimeError("The setup did not produce the intended row slices")

    stored_segments = len(layout)

    # First measure a direct full-table read to check how the counters behave for the known stored layout
    with collect_storage_stats() as baseline_stats:
        baseline = lib.read(
            symbol,
            as_of=1,
            output_format=OutputFormat.POLARS,
        ).data

    assert_frame_equal(baseline, expected)

    # Get the number of recorded table-data fetches
    baseline_count = query_stats_operation_count(baseline_stats, "Memory_GetObject", "TABLE_DATA")

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
        query = pc.scan_arcticdb(lib, symbol, as_of=1)

        # Discover column names and types before measuring collection
        query.collect_schema()

        # Record only the storage work performed during collection
        with collect_storage_stats() as scan_stats:
            actual = query.collect(engine="streaming")

    # Batching must preserve every row, its order and its type
    assert_frame_equal(actual, expected)

    # Get the data-fetch count for the Polarctic scan
    scan_count = query_stats_operation_count(scan_stats, "Memory_GetObject", "TABLE_DATA")

    # The complete scan needs the same stored segments as the reference read.
    # Its small output hint should not cause extra segment fetches
    assert scan_count == stored_segments, (
        f"Expected {stored_segments} data fetches, recorded {scan_count}"
    )


def test_projection_reduces_table_data_reads(stats_library, sample_data):
    """Check that selecting one column fetches fewer data segments."""
    # Get this test's library and symbol
    lib, symbol = stats_library

    data = sample_data
    lib.write(symbol, data)

    # Check the actual layout
    layout = data_segment_layout(lib, symbol, 0)
    column_ranges = layout[["start_col", "end_col"]].drop_duplicates()

    # Fail if the fixture did not create separate column segments
    if len(column_ranges) < 2:
        raise RuntimeError("The fixture did not create separate column segments")

    # Confirm that the storage counters record real data reads
    assert_data_reads_are_recorded(lib, symbol)

    # Compare a full read with a one-column read of the same version, holding the version and batch hint fixed
    with override_batch_hint(None):
        full_query = pc.scan_arcticdb(lib, symbol, as_of=0)
        narrow_query = pc.scan_arcticdb(lib, symbol, as_of=0).select("a")

        # Resolve schemas before measuring collection
        full_query.collect_schema()
        narrow_query.collect_schema()

        with collect_storage_stats() as full_stats:
            full = full_query.collect()

        with collect_storage_stats() as narrow_stats:
            narrow = narrow_query.collect()

    # First verify that both queries return the correct data
    assert_frame_equal(full, data)
    assert_frame_equal(narrow, data.select("a"))

    # Read the number of recorded table-data fetches for each query
    full_count = query_stats_operation_count(full_stats, "Memory_GetObject", "TABLE_DATA")
    narrow_count = query_stats_operation_count(narrow_stats, "Memory_GetObject", "TABLE_DATA")

    # The one-column query should fetch fewer data segments
    assert 0 < narrow_count < full_count, (
        f"Full read fetched {full_count} segments; "
        f"one-column read fetched {narrow_count}"
    )


# The current connector returns unfiltered rows for this time predicate
@pytest.mark.xfail(
    reason="Known bug: timestamp-index filter returns unfiltered rows",
    strict=True,
    raises=AssertionError,
)
def test_timeseries_index_filter_returns_correct_rows_and_skips_slices(stats_library):
    """Check correct time filtering and avoidance of out-of-range data fetches."""
    # Get this test's library and symbol
    lib, symbol = stats_library

    # Create one row per day, from January 1 through January 12.
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

    # Write separate three-day groups so the time range can exclude whole slices
    lib.write(symbol, data.slice(0, 3), index_column=True)

    for offset in [3, 6, 9]:
        lib.append(
            symbol,
            data.slice(offset, 3),
            index_column=True,
        )

    # One write and three appends create version 3
    version = 3

    # Read metadata describing the stored data segments before measuring the query 
    # so it is not counted as part of the query's collection
    layout = data_segment_layout(lib, symbol, version)

    # Choose a range covering January 4, 5 and 6
    start = dt.datetime(2020, 1, 4)
    end = dt.datetime(2020, 1, 7)

    predicate = (
        (pl.col("timestamp") >= start)
        & (pl.col("timestamp") < end)
    )
    expected = oracle(data, lambda frame: frame.filter(predicate))

    # A segment overlaps only if it starts before the requested end and ends after the requested start
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

    # First measure a direct full-table read to check how the counters behave for the known stored layout
    with collect_storage_stats() as full_stats:
        full = lib.read(
            symbol,
            as_of=version,
            output_format=OutputFormat.POLARS,
        ).data

    assert_frame_equal(full, data)

    # One fetch per data segment is expected
    assert query_stats_operation_count(full_stats, "Memory_GetObject", "TABLE_DATA") == len(layout)

    # Disable batching to focus on time filtering and storage skipping
    with override_batch_hint(None):
        query = pc.scan_arcticdb(lib, symbol, as_of=version).filter(predicate)

        # Discover column names and types before measuring collection
        query.collect_schema()

        # Record the read requests and tables returned by ArcticDB
        with observe_reads() as reads:
            # Record memory-storage reads during collection
            with collect_storage_stats() as filtered_stats:
                actual = query.collect()

    # Batching is disabled, so expect one executing backend read
    assert len(reads) == 1
    read = reads[0]

    # ArcticDB must already have returned the correctly filtered rows
    assert_frame_equal(read.data, expected)

    # The final Polarctic result must also be correct
    assert_frame_equal(actual, expected)

    # Check storage work separately: only the overlapping data segments should need to be fetched
    assert query_stats_operation_count(
        filtered_stats, "Memory_GetObject", "TABLE_DATA"
    ) == required_segments


def test_stats_enabled_does_not_change_results(stats_library, sample_data):
    """Check that collecting statistics does not change the query result."""
    # Get this test's library and symbol
    lib, symbol = stats_library
    lib.write(symbol, sample_data)
    assert_data_reads_are_recorded(lib, symbol)

    def apply(frame):
        # Use the same filter and column order for both executions
        return frame.filter(pl.col("a") > 1).select("b", "a")

    # Disable batching to keep the comparison focused on statistics collection
    with override_batch_hint(None):
        # Run a fresh query without enabling statistics
        without_stats = apply(pc.scan_arcticdb(lib, symbol, as_of=0)).collect()

        # Run the equivalent query with statistics enabled (both queries read the same stored version)
        with collect_storage_stats() as stats:
            with_stats = apply(pc.scan_arcticdb(lib, symbol, as_of=0)).collect()

    # Calculate the expected answer independently using only Polars
    expected = oracle(sample_data, apply)

    # Both executions must have the same correct values, order and types
    assert_frame_equal(without_stats, expected)
    assert_frame_equal(with_stats, expected)

    # Confirm that the measured execution actually recorded table-data reads
    assert query_stats_operation_count(stats, "Memory_GetObject", "TABLE_DATA") > 0


def test_querybuilder_date_range_skips_slices(stats_library):
    """Check whether QueryBuilder.date_range skips stored slices like read(date_range=...)."""
    # Get the managed S3 library and this test's unique symbol
    lib, symbol = stats_library

    # Twelve daily rows, January 1 to January 12, with the timestamp as the index
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

    # Write four separate three-day slices (one write and three appends)
    lib.write(symbol, data.slice(0, 3), index_column=True)
    for offset in [3, 6, 9]:
        lib.append(symbol, data.slice(offset, 3), index_column=True)
    version = 3

    # The time range we ask for: January 4 to January 6 (both included)
    start = pd.Timestamp("2020-01-04")
    end = pd.Timestamp("2020-01-07") - pd.Timedelta(1, "ns")
    expected = data.filter(pl.col("timestamp").is_between(start, end))

    # Count how many stored slices overlap that range (end_index is exclusive)
    layout = data_segment_layout(lib, symbol, version)
    overlaps = (layout.index <= end) & (layout["end_index"] > start)
    required = int(overlaps.sum())
    total = len(layout)
    if not 0 < required < total:
        raise RuntimeError("The setup must create slices both inside and outside the range")

    def count_reads(**read_kwargs):
        """Read once with the given arguments; return the rows and the slices fetched."""
        with collect_storage_stats() as stats:
            result = lib.read(
                symbol, as_of=version, output_format=OutputFormat.POLARS, **read_kwargs
            ).data
        return result, query_stats_operation_count(stats, "Memory_GetObject", "TABLE_DATA")

    # 1. Full read: shows that the counters work (expect one fetch per slice)
    full, full_count = count_reads()
    # 2. date_range as a read argument: the known way to skip slices
    by_argument, argument_count = count_reads(date_range=(start, end))
    # 3. date_range as a QueryBuilder step: the method we want to check
    by_builder, builder_count = count_reads(query_builder=QueryBuilder().date_range((start, end)))
    # 4. QueryBuilder date_range followed by row_range: what a streaming batch would use
    first_batch, batch_count = count_reads(
        query_builder=QueryBuilder().date_range((start, end)).row_range((0, 2))
    )

    # Print a small table (shown with pytest -s)
    print(f"\nslices stored: {total}, slices overlapping the range: {required}")
    print(f"full read:                          {full_count} slices fetched")
    print(f"read(date_range=...):               {argument_count} slices fetched")
    print(f"QueryBuilder().date_range(...):     {builder_count} slices fetched")
    print(f"QueryBuilder date_range + row_range: {batch_count} slices fetched")

    # The counters must work, and the known method must skip slices,
    # otherwise the comparison below means nothing
    assert full_count == total, "Counters did not record one fetch per slice"
    assert argument_count == required, "read(date_range=...) did not skip slices"

    # Every method must still return the right rows
    assert_frame_equal(full, data)
    assert_frame_equal(by_argument, expected)
    assert_frame_equal(by_builder, expected)
    assert_frame_equal(first_batch, expected.head(2))

    # The actual question: does the QueryBuilder step skip slices too?
    assert builder_count == required, (
        f"QueryBuilder.date_range fetched {builder_count} slices, expected {required}"
    )
    assert batch_count <= required, (
        f"QueryBuilder date_range + row_range fetched {batch_count} slices, expected at most {required}"
    )

