"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Tests 4: execute filters, rather than only compare constructed queries."""

import datetime as dt

import polars as pl
import pytest
from arcticdb import QueryBuilder
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc


# These expressions must return the correct answer, by running either in ArcticDB or, if unsupported there, in Polars.
# Integer bitwise operations are included for correctness. Their inclusion does not mean ArcticDB must support them natively
PREDICATE_CASES = [
    pytest.param(pl.col("a") > 2, id="gt"),
    pytest.param(pl.col("a") >= 2, id="ge"),
    pytest.param(pl.col("a") < 2, id="lt"),
    pytest.param(pl.col("a") <= 2, id="le"),
    pytest.param(pl.col("a") == 2, id="eq"),
    pytest.param(pl.col("a") != 2, id="ne"),
    pytest.param((pl.col("a") + pl.col("b")) > 14, id="add"),
    pytest.param((pl.col("b") - pl.col("a")) == 10, id="subtract"),
    pytest.param((pl.col("a") * 2) > 5, id="multiply"),
    pytest.param((pl.col("b") / 2) > 6, id="divide"),
    pytest.param((-pl.col("a")) < -2, id="negate"),
    pytest.param(pl.col("a").abs() > 2, id="abs"),
    pytest.param((pl.col("a") > 1) & (pl.col("b") < 15), id="and"),
    pytest.param((pl.col("a") < 1) | (pl.col("a") > 4), id="or"),
    pytest.param(~pl.col("flag"), id="not"),
    pytest.param((pl.col("a") & 1) == 0, id="bitwise-and"),
    pytest.param((pl.col("a") | 1) == 3, id="bitwise-or"),
    pytest.param((pl.col("a") ^ 1) == 3, id="bitwise-xor"),
    pytest.param(pl.col("a").is_in([1, 4]), id="membership"),
    pytest.param(pl.col("text").str.contains("alpha|beta"), id="regex"),
    pytest.param(pl.col("text") == 'a"b', id="quoted-literal"),
    pytest.param(pl.col("text") == "[x]", id="bracket-literal"),
]

@pytest.mark.parametrize("predicate", PREDICATE_CASES)
def test_predicate_matches_polars(scan_store, scan_factory, batch_hint, oracle, predicate):
    """Check that executing each filter translated to ArcticDB returns the same rows and types as Polars."""
    # Disable batching so this test focuses on filtering and excludes any streaming effects
    batch_hint(None)

    # Run the filter through the connector
    actual = scan_factory().filter(predicate).collect()

    # Independently run the same filter on the known input using only Polars
    expected = oracle(scan_store.expected, lambda frame: frame.filter(predicate))

    # Check rows, values, column order and data types, without establishing where the filtering happened
    assert_frame_equal(actual, expected)


# These expressions are required to run in ArcticDB for this test suite. This is not the full list the translator attempts to 
# support, e.g., integer bitwise operations have translation code but currently fail in ArcticDB.
# TO DO: Confirm the team's agreed MVP native-pushdown requirements ???
# Repeating expressions from PREDICATE_CASES is intentional: the first test checks the answer; the next test checks execution location
REQUIRED_NATIVE_PREDICATES = [
    pytest.param(pl.col("a") > 2, id="gt"),
    pytest.param(pl.col("a") >= 2, id="ge"),
    pytest.param(pl.col("a") < 2, id="lt"),
    pytest.param(pl.col("a") <= 2, id="le"),
    pytest.param(pl.col("a") == 2, id="eq"),
    pytest.param(pl.col("a") != 2, id="ne"),
    pytest.param(
        (pl.col("a") > 1) & (pl.col("b") < 15),
        id="and",
    ),
    pytest.param(
        (pl.col("a") < 1) | (pl.col("a") > 4),
        id="or",
    ),
]

@pytest.mark.parametrize("predicate", REQUIRED_NATIVE_PREDICATES)
def test_required_native_predicate_is_executed_by_arcticdb(
    scan_store,
    scan_factory,
    observed_reads,
    batch_hint,
    oracle,
    predicate,
):
    """Check that required native filters run in ArcticDB."""
    # Disable batching so this test focuses on filtering and excludes any streaming effects
    batch_hint(None)

    # Calculate the expected answer independently, using only Polars
    expected = oracle(
        scan_store.expected,
        lambda frame: frame.filter(predicate),
    )

    # Every chosen predicate must remove some rows. Otherwise an unfiltered
    # backend read could accidentally pass the backend equality check
    assert 0 < expected.height < scan_store.expected.height

    actual = scan_factory().filter(predicate).collect()

    # Check that real ArcticDB reads happened
    assert observed_reads, "No executing ArcticDB read was observed"

    # Check that every executing read received a nonempty ArcticDB query
    empty_query = QueryBuilder()
    for read in observed_reads:
        query = read.kwargs.get("query_builder")
        assert isinstance(query, QueryBuilder), (
            "The ArcticDB read did not receive a QueryBuilder"
        )
        assert query != empty_query, (
            "The ArcticDB read received an empty QueryBuilder"
        )

    # These tables were recorded directly after ArcticDB returned them,
    # before the connector or Polars could apply any remaining filtering
    backend_output = pl.concat(
        [pl.from_arrow(read.data) for read in observed_reads]
    )

    # ArcticDB itself must already have returned the correct filtered rows
    assert_frame_equal(backend_output, expected)

    # The final result must also contain the correct rows, order and types
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("failure_type", [NotImplementedError, ValueError])
def test_translation_failure_still_filters_rows(scan_store, scan_factory, batch_hint, monkeypatch, failure_type):
    """Check that translation failures still produce correctly filtered data through local fallback."""
    calls = []

    def fail(predicate, query_builder):
        # Record the attempt, then deliberately refuse translation
        calls.append(predicate)
        raise failure_type("unsupported predicate")

    # Replace translation only (real ArcticDB reading still happens).
    # monkeypatch restores the original translator after this test
    monkeypatch.setattr(pc._TRANSLATOR, "translate", fail)
    batch_hint(None)

    predicate = pl.col("a") > 2
    actual = scan_factory().filter(predicate).collect()

    # Check that we successfully triggered the translation error set up by this test
    assert calls, "The predicate never reached the translator"

    # Translation did fail failed, but the requested filter must still be applied.
    # The connector must use Polars to apply the filter, not return all rows or fail
    assert_frame_equal(actual, scan_store.expected.filter(predicate))


@pytest.mark.parametrize("condition", ["modulo", "and", "or", "literal-option", "date-method"])
def test_unsupported_predicate_falls_back_without_losing_conditions(
    scan_store, scan_factory, batch_hint, monkeypatch, condition
):
    """Check that unsupported expressions and mixed conditions remain effective when passed to the reader."""
    unsupported = (pl.col("a") % 2) == 0  # remainder arithmetic is unsupported by the current translator
    predicates = {
        "modulo": unsupported,

        # Keep rows where a is even AND b > 11
        "and": unsupported & (pl.col("b") > 11),

        # Keep rows where a is even OR equals 5 (filtering only for 5 in ArcticDB would wrongly remove the even numbers)
        "or": unsupported | (pl.col("a") == 5),

        # Match the exact text "[x]", not its regex meaning
        "literal-option": pl.col("text").str.contains("[x]", literal=True),

        # Keep rows whose timestamp's day of the month is greater than 3
        "date-method": pl.col("ts").dt.day() > 3,
    }

    predicate = predicates[condition]
    calls = []
    original = pc._TRANSLATOR.translate

    def trace(expr, query_builder):
        # Observe translation without changing its behaviour
        calls.append(expr)
        return original(expr, query_builder)

    # Temporarily use that wrapper for traslation attempts
    monkeypatch.setattr(pc._TRANSLATOR, "translate", trace)

    batch_hint(None)
    actual = scan_factory().filter(predicate).collect()

    # Check that the connector actually tried to translate this filter
    assert calls, "The expression remained above the scan; fallback was not exercised"

    assert_frame_equal(actual, scan_store.expected.filter(predicate))


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize("operation", ["filter_select", "filter_head", "head_filter", "calculate_filter"])
def test_fallback_preserves_dependencies_and_operation_order(scan_store, scan_factory, batch_hint, oracle, engine, operation):
    """Check difficult filtering combined with other Polars operations."""
    # Use small batches for streaming to exercise operation combinations
    # across batches. Use one read for the in-memory case
    batch_hint(3 if engine == "streaming" else None)
    predicate = (pl.col("a") % 2) == 0

    def apply(frame):
        if operation == "filter_select":
            # The filter needs "a", even though the result only contains "desk".
            # Removing "a" too early would prevent correct filtering
            return frame.filter(predicate).select("desk")
        
        if operation == "filter_head":
            # Find matching rows first, then keep the first two matches
            return frame.filter(predicate).head(2)
        
        if operation == "head_filter":
            # Keep the first two source rows first, then filter those rows (different operation order than the previous case)
            return frame.head(2).filter(predicate)

        # "c" is created by Polars (not a stored ArcticDB column). Calculate it before filtering it, then return only "c".
        # This also checks processing above the scan, not necessarily connector fallback, because Polars may retain this filter itself
        return frame.with_columns(c=pl.col("a") + 1).filter((pl.col("c") % 2) == 0).select("c")

    # Apply the same operation sequence to both inputs
    actual = apply(scan_factory()).collect(engine=engine)
    expected = oracle(scan_store.expected, apply)
    assert_frame_equal(actual, expected)


def test_time_range_and_unsupported_filter_return_correct_rows(scan_store, scan_factory, batch_hint, oracle):
    """Check that a time condition and a filter requiring Polars remain effective together."""
    batch_hint(None)

    # A row must satisfy both the timestamp condition and the numeric filter
    predicate = (pl.col("ts") >= dt.datetime(2020, 1, 3)) & ((pl.col("a") % 2) == 0)

    def apply(frame):
        return frame.filter(predicate).select("ts", "a")

    actual = apply(scan_factory()).collect()
    expected = oracle(scan_store.expected, apply)

    # Check the complete result, including timestamp types and row order.
    # Partial pushdown is not required: applying the whole filter in Polars
    # is acceptable when ArcticDB cannot execute the complete expression
    assert_frame_equal(actual, expected)