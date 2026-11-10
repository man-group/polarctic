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
from tests.utils import oracle

def test_scan_returns_lazy_frame_and_complete_schema(arctic_source, sample_data):
    """Check that the Library scan form returns a lazy plan with the expected schema."""
    # Write the test data into the temporary library
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Prepare a scan of the first stored version
    query = pc.scan_arcticdb(lib, "data", as_of=0)

    # Check that the connector returns a Polars lazy plan
    assert isinstance(query, pl.LazyFrame)

    # Check the planned column names, order and types
    assert query.collect_schema() == sample_data.schema

    # Check the connector's schema helper independently
    assert pc.parse_schema(lib, "data", as_of=0) == sample_data.schema


@pytest.mark.parametrize(
    "operation",
    ["read", "filter", "select", "filter_select", "calculate", "group"],
)
def test_entry_points_return_expected_polars_data(arctic_source, sample_data, operation):
    """Check complete Polars results for basic operations through the library API."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Apply the same Polars operation to the connector's output and the independent reference data
    def apply(frame):
        if operation == "filter":
            # Keep rows satisfying both conditions
            return frame.filter((pl.col("a") > 1) & (pl.col("b") < 15))
        if operation == "select":
            # Request columns in a different order from the stored table
            return frame.select("b", "a")
        if operation == "filter_select":
            # Filter using b, then return only desk and a
            return frame.filter(pl.col("b") > 12).select("desk", "a")
        if operation == "calculate":
            # Add a calculated column without removing the original columns
            return frame.with_columns(c=pl.col("a") + pl.col("b"))
        if operation == "group":
            # Calculate the average b for each desk, then sort so the expected row order is predictable
            return frame.group_by("desk").agg(pl.col("b").mean()).sort("desk")
        return frame

    # Run the selected operation through Polarctic and also through Polars independently
    query = pc.scan_arcticdb(lib, "data", as_of=0)
    actual = apply(query).collect()
    expected = oracle(sample_data, apply)

    # Check rows, values, column order and types (backend execution location is checked separately in the pushdown tests)
    assert_frame_equal(actual, expected)


def test_requested_version_is_preserved(arctic_source, sample_data):
    """Check that an explicit version selects the correct stored data."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    changed = sample_data.with_columns(a=pl.col("a") + 100)
    lib.write("data", changed)

    assert_frame_equal(
        pc.scan_arcticdb(lib, "data", as_of=0).collect(),
        sample_data,
    )
    assert_frame_equal(
        pc.scan_arcticdb(lib, "data", as_of=1).collect(),
        changed,
    )


def test_scan_validates_required_arguments(arctic_source):
    """Check that missing required arguments and unsupported source types are rejected clearly."""
    # URI input with missing symbol
    with pytest.raises(ValueError, match="lib_name and symbol are required"):
        pc.scan_arcticdb(arctic_source.uri, arctic_source.library_name)

    # Library input with a missing symbol
    with pytest.raises(ValueError, match="symbol is required"):
        pc.scan_arcticdb(arctic_source.library)

    # Unsupported source
    with pytest.raises(TypeError, match="Unsupported source type"):
        pc.scan_arcticdb(123)


def test_registration_normalizes_output_to_polars(arctic_source, sample_data):
    """Check that registration forces the output format used by the reader."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Prepare an ArcticDB lazy source using Polars output
    source = lib.read("data", as_of=0, lazy=True, output_format=OutputFormat.POLARS)

    # Deliberately change its read request to pandas output.
    # Registration must correct this before the reader executes
    request = source._to_read_request()._replace(
        output_format=OutputFormat.PANDAS
    )

    # Register the source with its schema getter and deliberately wrong request
    query = pc._register_arctic_source(lib, source._collect_schema, lambda: request)

    # Collection must succeed and preserve the expected values and schema
    assert_frame_equal(query.collect(), sample_data)


def test_uri_reads_real_store(tmp_path):
    """Check that the URI API opens a real database and reads its data."""
    # Create a separate temporary database for the URI entry-point test
    uri = f"lmdb://{tmp_path / 'uri_store'}"
    ac = Arctic(uri)
    lib = ac.create_library("uri_tests")

    expected = pl.DataFrame({"a": [0, 1, 2], "b": [10.0, 11.0, 12.0]})
    lib.write("data", expected)

    # Release the write connection before the connector opens the same database
    del lib, ac
    gc.collect()

    # Pass the actual URI, library name and symbol (must open the connection to source)
    actual = pc.scan_arcticdb(uri, "uri_tests", "data", as_of=0).collect()
    assert_frame_equal(actual, expected)