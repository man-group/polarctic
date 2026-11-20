"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Check filter execution rather than only compare constructed queries."""

import datetime as dt

import polars as pl
import pytest
from arcticdb import QueryBuilder
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import observe_reads, oracle, override_batch_hint

from arcticdb.exceptions import UserInputException


# Known failure groups:
# - INCORRECT_FILTER: arithmetic, column-dependent or quoted-string filters return incorrect rows.
# - INTEGER_BITWISE: ArcticDB rejects the translated integer AND/OR/XOR queries.
# - MISSING_FALLBACK: unsupported filters are lost instead of being applied in Polars.
# - LITERAL_OPTION: literal=True is ignored, so literal text is treated as a regex.


INCORRECT_FILTER = pytest.mark.xfail(
    reason="Known bug: the filter is ignored or produces incorrect rows",
    strict=True,
    raises=AssertionError,
)

INTEGER_BITWISE = pytest.mark.xfail(
    reason="Known bug: ArcticDB rejects the translated integer bitwise query",
    strict=True,
    raises=(pl.exceptions.ComputeError, UserInputException),
)

MISSING_FALLBACK = pytest.mark.xfail(
    reason="Known bug: unsupported predicates are not applied locally",
    strict=True,
    raises=AssertionError,
)

LITERAL_OPTION = pytest.mark.xfail(
    reason="Known bug: literal=True is not preserved during translation",
    strict=True,
    raises=AssertionError,
)


# These expressions must return the correct answer, by running either in ArcticDB or, if unsupported there, in Polars.
# Integer bitwise operations are included for correctness. Their inclusion does not mean ArcticDB must support them natively
PREDICATE_CASES = [
    pytest.param(pl.col("a") > 2, id="gt"),
    pytest.param(pl.col("a") >= 2, id="ge"),
    pytest.param(pl.col("a") < 2, id="lt"),
    pytest.param(pl.col("a") <= 2, id="le"),
    pytest.param(pl.col("a") == 2, id="eq"),
    pytest.param(pl.col("a") != 2, id="ne"),
    pytest.param(pl.col("a") > (pl.col("b") - 12), id="column-gt", marks=INCORRECT_FILTER),
    pytest.param(pl.col("a") == (pl.col("b") - 10), id="column-eq", marks=INCORRECT_FILTER),
    pytest.param((pl.col("a") + pl.col("b")) > 14, id="add", marks=INCORRECT_FILTER),
    pytest.param((pl.col("b") - pl.col("a")) == 10, id="subtract", marks=INCORRECT_FILTER),
    pytest.param((pl.col("a") * 2) > 5, id="multiply"),
    pytest.param((pl.col("b") / 2) > 6, id="divide"),
    pytest.param((-pl.col("a")) < -2, id="negate"),
    pytest.param(pl.col("a").abs() > 2, id="abs"),
    pytest.param((pl.col("a") > 1) & (pl.col("b") < 15), id="and"),
    pytest.param((pl.col("a") < 1) | (pl.col("a") > 4), id="or"),
    pytest.param(~pl.col("flag"), id="not"),
    pytest.param((pl.col("a") & 1) == 0, id="bitwise-and", marks=INTEGER_BITWISE),
    pytest.param((pl.col("a") | 1) == 3, id="bitwise-or", marks=INTEGER_BITWISE),
    pytest.param((pl.col("a") ^ 1) == 3, id="bitwise-xor", marks=INTEGER_BITWISE),
    pytest.param(pl.col("a").is_in([1, 4]), id="membership"),
    pytest.param(pl.col("text").str.contains("alpha|beta"), id="regex"),
    pytest.param(pl.col("text") == 'a"b', id="quoted-literal", marks=INCORRECT_FILTER),
    pytest.param(pl.col("text") == "[x]", id="bracket-literal"),
]

@pytest.mark.parametrize("predicate", PREDICATE_CASES)
def test_predicate_matches_polars(arctic_source, sample_data, predicate):
    """Check each filter's result, allowing native execution or Polars fallback."""
    # Write the test data into the temporary library
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Calculate the expected result independently using only Polars
    expected = oracle(sample_data, lambda frame: frame.filter(predicate))

    # Ensure this dataset contains both matching and nonmatching rows
    if not 0 < expected.height < sample_data.height:
        raise RuntimeError(
            "The test data must contain matching and nonmatching rows"
        )

    # Disable batching so this test focuses on filtering.
    # Create the scan inside the block so it receives this override
    with override_batch_hint(None):
        actual = (
            pc.scan_arcticdb(lib, "data", as_of=0)
            .filter(predicate)
            .collect()
        )

    # Check rows, values, column order and data types, without establishing where the filtering happened
    assert_frame_equal(actual, expected)


# These expressions are required to run in ArcticDB for this test suite. This is not the full list the translator attempts to 
# support, e.g., integer bitwise operations have translation code but currently fail in ArcticDB.
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
def test_required_native_predicate_is_executed_by_arcticdb(arctic_source, sample_data, predicate):
    """Check that required native filters run in ArcticDB."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Calculate the expected result independently using only Polars
    expected = oracle(sample_data, lambda frame: frame.filter(predicate))

    # Ensure this dataset contains both matching and nonmatching rows
    if not 0 < expected.height < sample_data.height:
        raise RuntimeError(
            "The test data must contain matching and nonmatching rows"
        )

    # Disable batching so this test focuses on filtering
    with override_batch_hint(None):
        # Record the backend requests and data returned by ArcticDB
        with observe_reads() as reads:
            actual = (
                pc.scan_arcticdb(lib, "data", as_of=0)
                .filter(predicate)
                .collect()
            )

    # Batching is disabled, so expect one executing read
    assert len(reads) == 1
    read = reads[0]

    # Check that ArcticDB received a nonempty query.
    query_builder = read.kwargs.get("query_builder")
    assert isinstance(query_builder, QueryBuilder)
    assert query_builder != QueryBuilder()

    # ArcticDB must already have returned the correctly filtered rows.
    assert_frame_equal(read.data, expected)

    # Check the final result too.
    assert_frame_equal(actual, expected)


@MISSING_FALLBACK
@pytest.mark.parametrize("failure_type", [NotImplementedError, ValueError])
def test_translation_failure_still_filters_rows(arctic_source, sample_data, monkeypatch, failure_type):
    """Check that translation failures still produce correctly filtered data through local fallback."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Record whether our deliberately failing translator was called
    calls = []

    def fail(predicate, query_builder):
        # Record the attempt, then simulate unsupported translation
        calls.append(predicate)
        raise failure_type("unsupported predicate")

    # Temporarily replace translation, but keep real database reads
    monkeypatch.setattr(pc._TRANSLATOR, "translate", fail)

    predicate = pl.col("a") > 2
    expected = sample_data.filter(predicate)

    # Ensure this dataset contains both matching and nonmatching rows
    if not 0 < expected.height < sample_data.height:
        raise RuntimeError(
            "The test data must contain matching and nonmatching rows"
        )

    # Disable batching and run the filter through the connector
    with override_batch_hint(None):
        actual = (
            pc.scan_arcticdb(lib, "data", as_of=0)
            .filter(predicate)
            .collect()
        )

    # Check that we actually triggered the translation error.
    # Otherwise, correct output would not establish connector fallback
    if not calls:
        pytest.fail("The deliberately failing translator was never called")

    # Translation failed, but only matching rows should be returned, i.e., the 
    # connector must arrange Polars filtering instead of losing the filter
    assert_frame_equal(actual, expected)



@pytest.mark.parametrize(
    "condition", 
    [
        pytest.param("modulo", marks=MISSING_FALLBACK),
        pytest.param("and", marks=MISSING_FALLBACK),
        pytest.param("or", marks=MISSING_FALLBACK),
        pytest.param("literal-option", marks=LITERAL_OPTION),
        pytest.param("date-method", marks=MISSING_FALLBACK),
    ],
)
def test_complex_predicates_preserve_all_conditions(arctic_source, sample_data, monkeypatch, condition):
    """Check that no unsupported expressions and mixed conditions are silently dropped."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Remainder arithmetic is unsupported by the current translator
    unsupported = (pl.col("a") % 2) == 0

    predicates = {
        # Keep rows where a is even
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

    # Pick the condition for this parameterised test run
    predicate = predicates[condition]
    expected = sample_data.filter(predicate)

    # Ensure this dataset contains both matching and nonmatching rows
    if not 0 < expected.height < sample_data.height:
        raise RuntimeError(
            "The test data must contain matching and nonmatching rows"
        )

    # Save the real translator and start a list of translation attempts
    calls = []
    original = pc._TRANSLATOR.translate

    def trace(expr, query_builder):
        # Record the expression, then use the real translator unchanged
        calls.append(expr)
        return original(expr, query_builder)

    # Temporarily send translation calls through our recording wrapper
    monkeypatch.setattr(pc._TRANSLATOR, "translate", trace)

    # Disable batching and execute the selected filter
    with override_batch_hint(None):
        actual = (
            pc.scan_arcticdb(lib, "data", as_of=0)
            .filter(predicate)
            .collect()
        )

    # Check that translation went through our recording wrapper
    if not calls:
        pytest.fail("The translator was never called")

    # Compare with the complete filter applied in Polars (every condition and option must remain effective)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
@pytest.mark.parametrize(
    "operation", 
    [
        pytest.param("filter_select", marks=MISSING_FALLBACK),
        pytest.param("filter_head", marks=MISSING_FALLBACK),
        "head_filter",
        "calculate_filter",
    ],
)
def test_fallback_preserves_dependencies_and_operation_order(arctic_source, sample_data, engine, operation):
    """Check filtering combined with selection, limits and calculations."""
    lib = arctic_source.library
    lib.write("data", sample_data)

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

        # Polars creates c (it is not a stored ArcticDB column). Calculate c before filtering it, then return only c.
        # Polars may keep this filter outside the connector, so this case checks combined processing rather than proving connector fallback
        return frame.with_columns(c=pl.col("a") + 1).filter((pl.col("c") % 2) == 0).select("c")

    # Check the filter itself, before any later limit or column selection
    if operation == "head_filter":
        # Here the filter receives only the first two source rows
        filter_input = sample_data.head(2)
        filtered = filter_input.filter(predicate)
    elif operation == "calculate_filter":
        # Here the filter receives the data after Polars creates c
        filter_input = sample_data.with_columns(c=pl.col("a") + 1)
        filtered = filter_input.filter((pl.col("c") % 2) == 0)
    else:
        # The other cases filter the complete input before limiting/selecting
        filter_input = sample_data
        filtered = filter_input.filter(predicate)

    # Ensure the filter keeps some input rows and removes others
    if not 0 < filtered.height < filter_input.height:
        raise RuntimeError(
            "The filter input must contain matching and nonmatching rows"
        )

    # Apply the same sequence to known input using only Polars
    expected = oracle(sample_data, apply)

    # Use small batches for streaming and one read for the in-memory case
    hint = 3 if engine == "streaming" else None

    with override_batch_hint(hint):
        query = apply(pc.scan_arcticdb(lib, "data", as_of=0))

        print(f"\nCASE: {operation}, engine={engine}")
        print("UNOPTIMIZED PLAN:")
        print(query.explain(optimized=False))
        print("OPTIMIZED PLAN:")
        print(query.explain(optimized=True))

        actual = query.collect(engine=engine)

    print("EXPECTED:")
    print(expected)
    print("ACTUAL:")
    print(actual)

    # Check that the complete operation sequence returns the correct result
    assert_frame_equal(actual, expected)


# Modulo is unsupported by the translator, so the connector currently loses this combined filter instead of applying it in Polars
@MISSING_FALLBACK
def test_time_range_and_unsupported_filter_return_correct_rows(arctic_source, sample_data):
    """Check that a time condition and numeric conditions remain effective."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # A row must satisfy both the timestamp condition and the numeric filter
    predicate = (
        (pl.col("ts") >= dt.datetime(2020, 1, 3))
        & ((pl.col("a") % 2) == 0)
    )

    def apply(frame):
        # Apply both conditions, then return only ts and a
        return frame.filter(predicate).select("ts", "a")

    # Calculate the expected answer independently using only Polars
    expected = oracle(sample_data, apply)

    # Ensure this dataset contains both matching and nonmatching rows
    if not 0 < expected.height < sample_data.height:
        raise RuntimeError(
            "The test data must contain matching and nonmatching rows"
        )

    # Disable batching to focus on the combined filter
    with override_batch_hint(None):
        query = pc.scan_arcticdb(lib, "data", as_of=0)
        actual = apply(query).collect()

    # Check that neither condition was lost (applying the complete filter using Polars fallback is acceptable)
    assert_frame_equal(actual, expected)