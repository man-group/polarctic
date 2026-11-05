"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Tests 2: observe real read requests and intermediate Arrow output."""

import polars as pl
from arcticdb import QueryBuilder
from polars.testing import assert_frame_equal


def backend_frame(reads):
    """Combine the tables returned directly by ArcticDB."""
    assert reads, "No executing ArcticDB read was observed"

    # Convert the recorded Arrow tables to Polars and join their rows.
    # These tables were recorded before any remaining local processing
    return pl.concat([pl.from_arrow(read.data) for read in reads])


def test_predicate_is_executed_by_arcticdb(scan_store, scan_factory, observed_reads, batch_hint):
    """Check that ArcticDB receives and executes the requested filter."""
    # Disable batching to focus on filtering and exclude any streaming effects
    batch_hint(None)

    actual = scan_factory().filter(pl.col("a") > 3).collect()
    expected = scan_store.expected.filter(pl.col("a") > 3)

    # Build the ArcticDB query we expect the connector to send
    qb = QueryBuilder()
    qb = qb[qb["a"] > 3]

    # Check that a real read happened and received the expected query
    assert observed_reads
    assert all(read.kwargs["query_builder"] == qb for read in observed_reads)

    # ArcticDB must already have returned only matching rows (a correct final result alone could hide filtering done only in Polars)
    assert_frame_equal(backend_frame(observed_reads), expected)
    assert_frame_equal(actual, expected)


def test_projection_is_executed_by_arcticdb(scan_store, scan_factory, observed_reads):
    """Check that ArcticDB is asked to return only the selected columns."""
    actual = scan_factory().select("a", "b").collect()

    assert observed_reads

    # Check which columns the connector requested from ArcticDB
    assert all(set(read.kwargs["columns"]) == {"a", "b"} for read in observed_reads)

    # Check that ArcticDB actually returned only those columns. 
    # Sets check column names here; final equality checks their order too
    assert all(set(read.data.column_names) == {"a", "b"} for read in observed_reads)

    # Check final values, column order and types
    assert_frame_equal(actual, scan_store.expected.select("a", "b"))


def test_filter_dependency_is_kept_until_filtering(scan_store, scan_factory, observed_reads, batch_hint):
    """Check that filtering can use a column omitted from the final result."""
    batch_hint(None)

    # The filter needs "b", but the user only wants "a" in the output
    actual = scan_factory().filter(pl.col("b") > 12).select("a").collect()
    expected = (scan_store.expected.filter(pl.col("b") > 12).select("a"))

    assert observed_reads

    # Check that the backend received a nonempty query (avoid exact query equality here because Polars may rewrite the filter)
    assert all(read.kwargs["query_builder"] != QueryBuilder() for read in observed_reads)

    # ArcticDB must have used "b" to return the correct matching "a" values
    assert_frame_equal(backend_frame(observed_reads).select("a"), expected)

    # The final result must be correct and must not include "b"
    assert_frame_equal(actual, expected)


def test_polars_calculation_is_not_added_to_backend_query(scan_store, scan_factory, observed_reads):
    """Check that Polars adds the calculated column after receiving data, not through backend pushdown."""
    actual = scan_factory().with_columns(c=pl.col("a") + pl.col("b")).collect()

    assert observed_reads

    # No calculation should have been added to the ArcticDB query
    assert all(read.kwargs["query_builder"] == QueryBuilder() for read in observed_reads)

    # ArcticDB should not return "c": Polars creates it afterwards
    assert all("c" not in read.data.column_names for read in observed_reads)

    # Check that the final result contains the correctly calculated column
    assert_frame_equal(actual, scan_store.expected.with_columns(c=pl.col("a") + pl.col("b")))


def test_polars_grouping_is_not_added_to_backend_query(scan_store, scan_factory, observed_reads):
    """Check that Polars groups input rows while the backend request contains no added grouping."""
    actual = scan_factory().group_by("desk").agg(pl.col("b").mean()).sort("desk").collect()

    assert observed_reads

    # No grouping should have been added to the ArcticDB query
    assert all(read.kwargs["query_builder"] == QueryBuilder() for read in observed_reads)

    # ArcticDB should return the input rows, not one summary row per desk
    assert backend_frame(observed_reads).height == scan_store.expected.height

    # Calculate the expected average for each desk using Polars (sorting makes the result order predictable for comparison)
    expected = scan_store.expected.group_by("desk").agg(pl.col("b").mean()).sort("desk")
    assert_frame_equal(actual, expected)


def test_planning_does_not_read_table_rows(scan_factory, observed_reads):
    """Check that building the query and discovering its schema do not execute table-row reads."""
    # Build the query and ask for its column names and types
    query = scan_factory().filter(pl.col("a") > 1).select("a")
    assert query.collect_schema() == pl.Schema({"a": pl.Int64})

    # Check that no non-lazy Library.read call has been recorded
    assert not observed_reads

    # Execute the query. A real read should now be recorded
    query.collect()
    assert observed_reads