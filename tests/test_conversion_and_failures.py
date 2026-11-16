"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Check returned types, missing values, and useful failures."""

import datetime as dt

import pandas as pd
import polars as pl
import pytest
from arcticdb import OutputFormat
from arcticdb.exceptions import ArcticException, NoSuchVersionException
from arcticdb.version_store.library import Library
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import oracle, override_batch_hint


def test_numeric_widths_and_special_values_are_preserved(arctic_source):
    """Check exact numeric types and values through a real stored read."""
    # Give every column an explicit type
    data = pl.DataFrame({
        "small": pl.Series([-128, 0, 127], dtype=pl.Int8),
        "unsigned": pl.Series([0, 1, 255], dtype=pl.UInt8),
        "large": pl.Series([-(2**63), 2**53 + 1, 2**63 - 1], dtype=pl.Int64),
        "single": pl.Series([1.5, float("nan"), -2.5], dtype=pl.Float32),
        "double": pl.Series([float("inf"), float("nan"), -float("inf")], dtype=pl.Float64),
    })

    lib = arctic_source.library
    lib.write("types", data)

    actual = pc.scan_arcticdb(lib, "types", as_of=0).collect()

    # Compare directly in Polars, requiring exact numeric values and types (e.g., 2**53 + 1 would lose precision if converted to float64)
    assert_frame_equal(actual, data, check_exact=True)


@pytest.mark.parametrize(
    "predicate",
    [
        # Keep rows where text is missing: a values 1 and 3
        pytest.param(pl.col("text").is_null(), id="null"),
            
        # Keep rows where text is present: a values 0 and 2
        pytest.param(
            ~pl.col("text").is_null(),
            id="not-null",
            marks=pytest.mark.xfail(
                reason="Known translator bug: negated null filter returns unfiltered rows",
                strict=True, 
                raises=AssertionError,
            ),
        ),
    ]
)
def test_null_string_filters_match_polars(arctic_source, predicate):
    """Check that null and non-null filters keep the correct rows."""
    # Create test data and write it into the temporary library
    data = pl.DataFrame(
        {
            "a": [0, 1, 2, 3],
            "text": ["alpha", None, "beta", None],
        }
    )

    lib = arctic_source.library
    lib.write("nulls", data)

    # Calculate the expected answer independently using only Polars
    expected = oracle(data, lambda frame: frame.filter(predicate))

    # Ensure the filter keeps some rows and removes others
    if not 0 < expected.height < data.height:
        raise RuntimeError(
            "The test data must contain matching and nonmatching rows"
        )

    # Disable batching to focus on filtering and exclude any streaming effects
    with override_batch_hint(None):
        actual = (
            pc.scan_arcticdb(lib, "nulls", as_of=0)
            .filter(predicate)
            .collect()
        )

    # Check complete rows and types
    assert_frame_equal(actual, expected)



@pytest.mark.parametrize("hint", [None, 3], ids=["single-read", "batched"])
@pytest.mark.parametrize(
    "operation",
    [
        # Read all rows and columns
        "read",

        # Keep rows where text is missing
        "null",

        # Keep rows where text is present (produced incorrect results in the recorded run)
        pytest.param(
            "not-null",
            marks=pytest.mark.xfail(
                reason="Known negated-null translation bug",
                strict=True,
                raises=AssertionError,
            ),
        ),

        # Keep rows where text equals "x"
        "equals-x",
    ],
)
def test_sparse_string_columns_across_appended_segments(
    arctic_source, request, hint, operation
):
    """Check reading and filtering a column absent from earlier stored segments."""
    lib = arctic_source.library

    # First write: the table contains only column a; text does not exist yet
    first_rows = pl.DataFrame({"a": [0, 1, 2]})

    lib.write("sparse", first_rows)

    # Append a column that did not exist in the first write (so its values in the earlier rows should be null)
    later_rows = pl.DataFrame(
        {
            "a": [3, 4, 5],
            "text": ["x", None, "y"],
        }
    )
    lib.append("sparse", later_rows)

    # Describe the complete expected table after the append (the first three rows have null text because 
    # the column did not exist; the last three rows contain the values supplied in the append)
    expected_data = pl.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "text": pl.Series(
                [None, None, None, "x", None, "y"],
                dtype=pl.String,
            ),
        }
    )

    # Define the operation once so both the scan and reference data
    # receive exactly the same filtering and selection
    def apply(frame):
        if operation == "read":
            return frame

        # Choose the filter for this test run.
        predicates = {
            # Missing text: a values 0, 1, 2 and 4
            "null": pl.col("text").is_null(),

            # Present text: a values 3 and 5
            "not-null": ~pl.col("text").is_null(),

            # The string "x": a value 3
            "equals-x": pl.col("text") == "x",
        }

        # Filter using text, but return only a (the reader must keep text available until filtering is finished)
        return frame.filter(predicates[operation]).select("a")

    # Calculate the expected result before collecting through Polarctic
    expected = oracle(expected_data, apply)

    # The read case deliberately returns the complete table.
    # Only the filtering cases should remove rows
    if operation != "read":
        if not 0 < expected.height < expected_data.height:
            raise RuntimeError(
                "The sparse test data must contain matching and nonmatching rows"
            )

    # With a three-row hint, the first read contains only rows without text.
    # None matches text == "x", but a matching row exists in the next group.
    # The current batching bug stops before reaching that later match
    if operation == "equals-x" and hint == 3:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Known B2 bug: matching strings occur after an empty filtered slice",
                strict=True,
                raises=AssertionError,
            )
        )

    # Version 1 includes both the original and appended rows
    with override_batch_hint(hint):
        query = pc.scan_arcticdb(lib, "sparse", as_of=1)

        actual = apply(query).collect(engine="streaming")

    # Check all values, row order, column order and types
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("operation", ["head_zero", "no_matches", "empty_symbol"])
def test_empty_results_keep_schema(arctic_source, sample_data, operation):
    """Check that empty results retain their column names, order and types."""
    lib = arctic_source.library

    # Create an empty reference table without losing its column types.
    # head(0) removes all rows but keeps the schema
    expected = sample_data.head(0)

    if operation == "empty_symbol":
        # Start with an empty stored table with the full explicitly typed schema
        lib.write("empty", expected)
        query = pc.scan_arcticdb(lib, "empty", as_of=0)
        # Compare with the empty-table representation returned directly by ArcticDB.
        # This checks that Polarctic preserves the backend output
        expected = lib.read("empty", as_of=0, output_format=OutputFormat.POLARS).data

    else:
        # Start with a nonempty stored table
        lib.write("data", sample_data)
        query = pc.scan_arcticdb(lib, "data", as_of=0)

        # One case requests no rows (just the expected columns); the other removes every row by filtering
        query = query.head(0) if operation == "head_zero" else query.filter(pl.col("a") > 100)

    # Before execution, the planned column names and types should be correct
    assert query.collect_schema() == expected.schema

    # Execute the query and check the complete empty result.
    # Having zero rows must not cause columns or their types to disappear
    assert_frame_equal(query.collect(), expected)


def test_missing_symbol_propagates_arcticdb_error(arctic_source):
    """Check that a missing symbol produces an error, not an empty table."""
    # The library is deliberately empty: no setup write is needed

    # Polars may wrap the error in ComputeError, but the ArcticDB error code should remain visible
    with pytest.raises((NoSuchVersionException, pl.exceptions.ComputeError), match="E_NO_SUCH_VERSION"):
        # The missing symbol may be detected when creating the scan or collecting it. 
        # Keep both steps inside the block so either error is caught
        pc.scan_arcticdb(arctic_source.library, "missing", as_of=0).collect()


def test_invalid_column_reports_polars_error(arctic_source, sample_data):
    """Check that requesting an unknown column produces an informative error."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Require the appropriate Polars error and the requested column name in its message
    with pytest.raises(pl.exceptions.ColumnNotFoundError, match="missing_column"):
        pc.scan_arcticdb(lib, "data", as_of=0).select("missing_column").collect()


@pytest.mark.parametrize("representation", ["recursive", "pickle"])
def test_unsupported_stored_representation_reports_clear_error(arctic_source, representation):
    """Check informative errors for stored objects that are not normal tables."""
    lib = arctic_source.library

    if representation == "recursive":
        # Store a dataframe inside a dictionary (ArcticDB can store this nested structure, but Polarctic
        # cannot scan it). Keep pandas format here to reproduce the original recursive-normalisation case
        lib.write("unsupported", {"table": pd.DataFrame({"a": [1, 2]})}, recursive_normalizers=True)
    else:
        # Store a saved Python dictionary
        lib.write_pickle("unsupported", {"a": [1, 2]})

    # Scanning these objects should raise an informative unsupported-data error,
    # indicating an error type and a relevant word in its message
    with pytest.raises(
        (ArcticException, NotImplementedError, pl.exceptions.ComputeError),
        match="(?i)(recursive|pickl|unsupported|not supported|schema)",
    ):
        pc.scan_arcticdb(lib, "unsupported", as_of=0).collect()


def test_read_failure_after_first_batch_is_not_silent(arctic_source, sample_data):
    """Check that a read failure stops collection instead of returning partial data."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Save the real read method and start a read counter
    original = Library.read
    count = 0

    def fail_second_read(self, *args, **kwargs):
        # Update the counter defined outside this function
        nonlocal count

        # Ignore lazy setup and only count executing reads from this library
        if self is lib and not kwargs.get("lazy", False):
            count += 1

            # Deliberately fail the second executing read
            if count == 2:
                raise RuntimeError("injected storage failure")

        return original(self, *args, **kwargs)

    # Temporarily use our wrapper instead of the normal read method
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Library, "read", fail_second_read)

        # Use a small batch hint so collection needs multiple reads.
        # batches records the data delivered before the failure
        with override_batch_hint(3) as batches:
            # Collection must report our error, not return a partial result
            with pytest.raises((RuntimeError, pl.exceptions.PolarsError), match="injected storage failure"):
                pc.scan_arcticdb(lib, "data", as_of=0).collect(engine="streaming")

    # Check that the second read was reached
    assert count == 2
    # Check that some data was delivered before the error
    assert batches


def test_timezone_timestamp_column_matches_stored_polars(arctic_source):
    """Check that Polarctic returns timestamps as ArcticDB returns them."""
    data = pl.DataFrame({
        "a": [0, 1, 2],
        "ts": pl.datetime_range(
            dt.datetime(2020, 7, 1),
            dt.datetime(2020, 7, 3),
            interval="1d", time_unit="ns", time_zone="Europe/London", eager=True,
        ),
    })
    lib = arctic_source.library
    lib.write("timezone", data)

    # Read version 0 directly from ArcticDB to get the reference result
    expected = lib.read("timezone", as_of=0, output_format=OutputFormat.POLARS).data

    # Read the same version through Polarctic
    actual = pc.scan_arcticdb(lib, "timezone", as_of=0).collect()

    # Check that Polarctic preserves the timestamp values and types returned by ArcticDB
    assert_frame_equal(actual, expected)


def test_all_null_string_result_keeps_string_type(arctic_source):
    """Check that a string column stays String when its returned values are null."""
    lib = arctic_source.library

    # Store two rows containing actual strings
    lib.write("all_null", (pl.DataFrame({"a": [0, 1], "text": ["x", "y"]})))

    # Append two rows with missing strings (specify String type explicitly: otherwise Polars infers Null type)
    lib.append("all_null", (pl.DataFrame({
        "a": [2, 3],
        "text": pl.Series([None, None], dtype=pl.String),
    })))

    # The expected text values are null, but the column type is String
    expected = pl.DataFrame({
        "a": [2, 3],
        "text": pl.Series([None, None], dtype=pl.String),
    })

    # Start at row position 2 and take the two appended rows only (use version 1)
    actual = pc.scan_arcticdb(lib, "all_null", as_of=1).slice(2, 2).collect()
    # Check the values and types, including String rather than Null
    assert_frame_equal(actual, expected)