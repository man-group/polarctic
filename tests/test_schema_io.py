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
from arcticdb import LibraryOptions, OutputFormat
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import collect_storage_stats, observe_reads, override_batch_hint, query_stats_operation_count


@pytest.fixture
def stats_library(library_factory):
    """Create an empty, isolated in-memory library for each test."""
    return library_factory(
        storage="memory",
        library_options=LibraryOptions(
            rows_per_segment=3,
            columns_per_segment=2,
        ),
    ).library


def data_segment_layout(lib, symbol, version=None):
    """Return the stored data-segment layout for a symbol version."""
    return lib._nvs.read_index(symbol, as_of=version)



def test_schema_discovery_reads_no_table_data_keys(stats_library, sample_data):
    """Check that planning reads no table-data segments."""
    # Get this test's library
    lib = stats_library
    lib.write("data", sample_data)

    # Record from scan creation onwards (this includes any schema lookup triggered while adding the operations)
    with collect_storage_stats() as planning:
        query = pc.scan_arcticdb(lib, "data").filter(pl.col("a") > 1).select("a")
        assert query.collect_schema() == pl.Schema({"a": pl.Int64})

    # Planning should not fetch the stored table-data segments
    assert query_stats_operation_count(planning, "Memory_GetObject", "TABLE_DATA") == 0

    # Fetching the index during planning is expected in this setup.
    # A positive count also confirms that this phase recorded index IO
    assert query_stats_operation_count(planning, "Memory_GetObject", "TABLE_INDEX") == 1


@pytest.mark.xfail(
    reason="Known bug: collection reloads the index obtained during schema discovery",
    strict=True,
    raises=AssertionError,
)
@pytest.mark.parametrize("hint", [None, 3], ids=["single-read", "batched"])
def test_collection_reuses_schema_index(stats_library, sample_data, hint):
    """Check index reuse during single-read and batched collection."""
    # Get this test's library
    lib = stats_library
    lib.write("data", sample_data)

    # Create the scan with the selected hint, and record the batches delivered by the reader
    with override_batch_hint(hint) as batches:

        # Measure planning separately from collection
        with collect_storage_stats() as planning:
            query = pc.scan_arcticdb(lib, "data")
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
    assert query_stats_operation_count(planning, "Memory_GetObject", "TABLE_INDEX") == 1

    # Collection should have read actual table data
    assert query_stats_operation_count(collection, "Memory_GetObject", "TABLE_DATA") > 0

    # The index obtained during planning should be reused during collection
    assert query_stats_operation_count(collection, "Memory_GetObject", "TABLE_INDEX") == 0


# Small batch hints currently cause extra fetches of stored table data
@pytest.mark.xfail(
    reason="Known bug: small batch hints cause repeated table-data fetches",
    strict=True,
    raises=AssertionError,
)
@pytest.mark.parametrize("hint", [1, 2, 3, 4])
def test_small_hints_do_not_repeat_table_data_fetches(stats_library, hint):
    """Check that small batch hints do not cause unnecessary data fetches."""
    # Get this test's library
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

    # The five-row write creates slices of 3 and 2 rows.
    # The five-row append creates another 3-row and 2-row slice
    assert layout["start_row"].tolist() == [0, 3, 5, 8]
    assert layout["end_row"].tolist() == [3, 5, 8, 10]

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

    # The complete scan needs four stored data segments.
    # A small batch hint must not fetch any of them again
    assert query_stats_operation_count(scan_stats, "Memory_GetObject", "TABLE_DATA") == 4


def test_projection_reduces_table_data_reads(stats_library, sample_data):
    """Check that selecting one column fetches fewer data segments."""
    # Get this test's library
    lib = stats_library

    data = sample_data
    lib.write("data", data)

    # Compare a full read with a one-column read of the same version, holding the version and batch hint fixed
    with override_batch_hint(None):
        full_query = pc.scan_arcticdb(lib, "data")
        narrow_query = pc.scan_arcticdb(lib, "data").select("a")

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

    # Eight rows create three row slices.
    # Six columns create three two-column groups: 3 × 3 = 9 data segments
    assert full_count == 9

    # Column a belongs to one column group across the three row slices
    assert narrow_count == 3


@pytest.mark.parametrize(
    ("start_day", "end_day", "expected_reads"),
    [
        # January 4–6: read one stored three-day group
        (4, 7, 1),

        # January 4–9: read two stored three-day groups
        (4, 10, 2),
    ],
)
def test_preapplied_date_range_skips_unneeded_slices(stats_library, start_day, end_day, expected_reads):
    """Preserve an ArcticDB date range and fetch only the needed data slices."""
    lib = stats_library

    # Create twelve rows: one for each day from January 1 to January 12
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

    # Store the rows in four separate groups: January 1–3, 4–6, 7–9 and 10–12
    lib.write("data", data.slice(0, 3), index_column=True)
    for offset in [3, 6, 9]:
        lib.append("data", data.slice(offset, 3), index_column=True)

    # One write and three appends create version 3, so use that version for this test
    start = dt.datetime(2020, 1, start_day)
    stop = dt.datetime(2020, 1, end_day)

    # ArcticDB's date-range end is inclusive so subtract one nanosecond to exclude the start of end_day
    end = pd.Timestamp(stop) - pd.Timedelta(1, "ns")

    # Calculate the expected rows directly from the original Polars table
    expected = data.filter(
        (pl.col("timestamp") >= start)
        & (pl.col("timestamp") < stop)
    )

    # Give ArcticDB the date range before creating the Polarctic scan (as this test focuses on 
    # preservation of an existing range, not translation of a Polars timestamp predicate)
    source = lib.read("data", as_of=3, date_range=(start, end), lazy=True, output_format=OutputFormat.POLARS)

    # Disable batching so the scan performs one executing backend read
    with override_batch_hint(None):
        query = pc.scan_arcticdb(source)

        # Discover the schema before measuring data collection
        query.collect_schema()

        # Record both the backend result and its storage operations
        with observe_reads() as reads, collect_storage_stats() as stats:
            actual = query.collect()

    # Exactly one backend read should have executed
    assert len(reads) == 1

    # ArcticDB itself must return only the requested dates
    assert_frame_equal(reads[0].data, expected)

    # The final Polarctic result must contain those same rows and types
    assert_frame_equal(actual, expected)

    # Only the stored groups inside the requested range should be fetched.
    # Checking the exact count catches unnecessary data-segment reads
    assert query_stats_operation_count(stats, "Memory_GetObject", "TABLE_DATA") == expected_reads