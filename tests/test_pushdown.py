"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Tests 2: check ArcticDB read requests and intermediate Polars results."""

import polars as pl
from arcticdb import QueryBuilder
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import backend_frame, observe_reads, override_batch_hint


def test_predicate_is_executed_by_arcticdb(arctic_source, sample_data):
    """Check the backend query and already-filtered backend result."""
    # Write the test data into the temporary library
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Build the ArcticDB query and the expected result in Polars
    qb = QueryBuilder()
    qb = qb[qb["a"] > 3]
    expected = sample_data.filter(pl.col("a") > 3)

    # Disable batching to focus on filtering and exclude any streaming effects
    with override_batch_hint(None):
        # Record requests and returned data while running the query
        with observe_reads() as reads:
            actual = (
                pc.scan_arcticdb(lib, "data", as_of=0)
                .filter(pl.col("a") > 3)
                .collect()
            )
    
    # Check that a real read happened and received the expected query
    assert reads
    assert all(read.kwargs["query_builder"] == qb for read in reads)

    # ArcticDB must already have returned only matching rows (a correct final result alone could hide filtering done only in Polars)
    assert_frame_equal(backend_frame(reads), expected)
    assert_frame_equal(actual, expected)


def test_projection_is_executed_by_arcticdb(arctic_source, sample_data):
    """Check requested and returned columns."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Record requests and returned data while running the query
    with observe_reads() as reads:
        actual = (
            pc.scan_arcticdb(lib, "data", as_of=0)
            .select("a", "b")
            .collect()
        )

    # Check that at least one executing ArcticDB read was recorded
    assert reads

    # Check which columns the connector requested from ArcticDB
    assert all(set(read.kwargs["columns"]) == {"a", "b"} for read in reads)

    # Check that ArcticDB actually returned only those columns
    assert all(set(read.data.columns) == {"a", "b"} for read in reads)

    # Check final values, column order and types
    assert_frame_equal(actual, sample_data.select("a", "b"))


def test_filter_dependency_is_kept_until_filtering(arctic_source, sample_data):
    """Check that filtering can use a column omitted from the final result."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Calculate the expected result directly in Polars
    expected = sample_data.filter(pl.col("b") > 12).select("a")

    # Disable batching to focus on filtering and exclude any streaming effects
    with override_batch_hint(None):
        # Record the backend requests and returned data
        with observe_reads() as reads:
            # The filter needs b, but the user only wants a in the output
            actual = (
                pc.scan_arcticdb(lib, "data", as_of=0)
                .filter(pl.col("b") > 12)
                .select("a")
                .collect()
            )

    # Check that an executing ArcticDB read was recorded
    assert reads

    # Check that ArcticDB received a nonempty query (avoid exact query equality because Polars may rewrite the filter)
    assert all(read.kwargs["query_builder"] != QueryBuilder() for read in reads)

    # ArcticDB must already have returned the correct matching a values
    assert_frame_equal(backend_frame(reads).select("a"), expected)

    # Check final values, column order and types (equality also checks that b is not included in the final output)
    assert_frame_equal(actual, expected)


def test_polars_calculation_is_not_added_to_backend_query(arctic_source, sample_data):
    """Check that Polars creates the calculated column after receiving backend data, and does not use backend pushdown."""
    lib = arctic_source.library
    lib.write("data", sample_data)
    
    # Record the backend requests and returned data
    with observe_reads() as reads:
        actual = (
            pc.scan_arcticdb(lib, "data", as_of=0)
            .with_columns(c=pl.col("a") + pl.col("b"))
            .collect()
        )

    # Check that an executing ArcticDB read was recorded
    assert reads

    # The backend query should be empty (no calculation was sent to ArcticDB)
    assert all(read.kwargs["query_builder"] == QueryBuilder() for read in reads)

    # ArcticDB should not return c (Polars creates it afterwards)
    assert all("c" not in read.data.columns for read in reads)

    # Check that the final result includes the correct calculated values, column order and types
    assert_frame_equal(actual, sample_data.with_columns(c=pl.col("a") + pl.col("b")))


def test_polars_grouping_is_not_added_to_backend_query(arctic_source, sample_data):
    """Check that grouping remains in Polars while the backend request contains no added grouping."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Record the backend requests and returned data
    with observe_reads() as reads:
        actual = (
            pc.scan_arcticdb(lib, "data", as_of=0)
            .group_by("desk")
            .agg(pl.col("b").mean())
            .sort("desk")
            .collect()
        )

    # Check that an executing ArcticDB read was recorded
    assert reads

    # The backend query should be empty (no grouping was sent to ArcticDB)
    assert all(read.kwargs["query_builder"] == QueryBuilder() for read in reads)

    # ArcticDB should return the input rows, not one summary row per desk
    assert backend_frame(reads).height == sample_data.height

    # Calculate the expected average for each desk directly in Polars,
    # then sort by desk so the comparison uses a predictable row order.
    expected = (
        sample_data.group_by("desk")
        .agg(pl.col("b").mean())
        .sort("desk")
    )

    # Check final values, column order and types
    assert_frame_equal(actual, expected)


def test_planning_does_not_read_table_rows(arctic_source, sample_data):
    """Check that building the query and discovering its schema do not execute table-row reads."""
    # Write the data before recording, so setup is not part of the observation
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Record only the reads made while building the query and discovering its schema
    with observe_reads() as planning_reads:
        query = (
            pc.scan_arcticdb(lib, "data", as_of=0)
            .filter(pl.col("a") > 1)
            .select("a")
        )

        # Check the planned output's column name and type, without collecting rows
        assert query.collect_schema() == pl.Schema({"a": pl.Int64})

    # Planning should not have triggered a recorded non-lazy Library.read call
    assert not planning_reads

    # Start a separate recording block and execute the query
    with observe_reads() as collection_reads:
        query.collect()

    # Now collection should have triggered at least one executing read
    assert collection_reads

    # This observer watches Library.read calls only. It does not prove that planning read 
    # no table-data storage keys through lower-level ArcticDB functions.