"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source 
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Tests 1: public entry points and complete Polars results."""

import gc
import polars as pl
import pytest
from arcticdb import OutputFormat, Arctic
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc

def test_scan_returns_lazy_frame_and_complete_schema(scan_store, scan_factory):
    """Check that each scan form returns a lazy plan with the complete expected schema."""
    query = scan_factory()
    assert isinstance(query, pl.LazyFrame)
    assert query.collect_schema() == scan_store.expected.schema
    assert pc.parse_schema(scan_store.library, "data", as_of=0) == scan_store.expected.schema


@pytest.mark.parametrize("operation", ["read", "filter", "select", "filter_select", "calculate", "group"])
def test_entry_points_return_expected_polars_data(scan_store, scan_factory, oracle, operation):
    """Check complete Polars results for each entry point and each requested operation."""
    # Apply the same Polars operation to the connector's output and the independent reference data
    def apply(frame):
        if operation == "filter":
            return frame.filter((pl.col("a") > 1) & (pl.col("b") < 15))
        if operation == "select":
            return frame.select("b", "a")
        if operation == "filter_select":
            return frame.filter(pl.col("b") > 12).select("desk", "a")
        if operation == "calculate":
            return frame.with_columns(c=pl.col("a") + pl.col("b"))
        if operation == "group":
            return frame.group_by("desk").agg(pl.col("b").mean()).sort("desk")
        return frame

    assert_frame_equal(apply(scan_factory()).collect(), oracle(scan_store.expected, apply))


def test_requested_version_is_preserved(scan_store, scan_factory):
    """Check that requesting an older or newer version returns that version through each scan form."""
    changed = scan_store.expected.with_columns(a=pl.col("a") + 100)
    scan_store.library.write("data", changed.to_pandas())
    assert_frame_equal(scan_factory(as_of=0).collect(), scan_store.expected)
    assert_frame_equal(scan_factory(as_of=1).collect(), changed)


def test_scan_validates_required_arguments(scan_store):
    """Check that missing required arguments and unsupported source types are rejected clearly."""
    with pytest.raises(ValueError, match="lib_name and symbol are required"):
        pc.scan_arcticdb(scan_store.uri, scan_store.library_name)
    with pytest.raises(ValueError, match="symbol is required"):
        pc.scan_arcticdb(scan_store.library)
    with pytest.raises(TypeError, match="Unsupported source type"):
        pc.scan_arcticdb(123)


def test_registration_normalizes_output_to_arrow(scan_store):
    """Check that source registration converts a pandas-output request to the required Arrow output."""
    source = scan_store.library.read("data", as_of=0, lazy=True, output_format=OutputFormat.PYARROW)
    # Deliberately supply the wrong output format to check registration normalisation
    request = source._to_read_request()._replace(output_format=OutputFormat.PANDAS)
    query = pc._register_arctic_source(
        scan_store.library, source._collect_schema, lambda: request
    )
    assert_frame_equal(query.collect(), scan_store.expected)


def test_uri_reads_real_store(tmp_path):
    """Check that the URI form opens the database and returns the expected data."""
    uri = f"lmdb://{tmp_path / 'uri_store'}"
    ac = Arctic(uri)
    lib = ac.create_library("uri_tests")
    expected = pl.DataFrame({"a": [0, 1, 2], "b": [10.0, 11.0, 12.0]})
    lib.write("data", expected.to_pandas())

    # Release the write connection before the connector opens the same database
    del lib, ac
    gc.collect()

    actual = pc.scan_arcticdb(uri, "uri_tests", "data", as_of=0).collect()
    assert_frame_equal(actual, expected)