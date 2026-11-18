"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Check complete batched results, filtering, and limit order."""

import polars as pl
import pytest
from arcticdb import Arctic, LibraryOptions, OutputFormat, QueryBuilder
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import backend_frame, observe_reads, oracle, override_batch_hint


# These cases reproduce the early termination bug after filtered reads.
# Assertion failures are expected; other exception types still fail normally
FILTERED_BATCHING = pytest.mark.xfail(
    reason="Known B2 bug: short or empty filtered reads stop batching early",
    strict=True,
    raises=AssertionError,
)


def combine(batches, schema):
    """Join the batches into one table, or create a correctly typed empty table."""
    # If batches were emitted, join their rows in the order received.
    # Otherwise, create an empty table with the expected columns and types
    return pl.concat(batches) if batches else pl.DataFrame(schema=schema)


@pytest.mark.parametrize("hint", [1, 2, 3, 4, 20])
def test_output_batches_preserve_all_rows(lmdb_tiny_segment_source, sample_data, hint):
    """Check that different batch hints preserve the complete table."""
    lib = lmdb_tiny_segment_source.library
    lib.write("data", sample_data)

    # Temporarily give newly created scans this batch-size hint.
    # Save each batch the reader produces in the batches list
    with override_batch_hint(hint) as batches:
        actual = pc.scan_arcticdb(
            lib, "data", as_of=0
        ).collect(engine="streaming")

    # Nonempty input should produce at least one batch
    assert batches, "No batches were emitted for nonempty input"

    # The current reader returns hint-sized batches, so only the last batch may be smaller.
    # Example: eight rows with hint=3 gives heights [3, 3, 2]
    expected_heights = [min(hint, sample_data.height - start) for start in range(0, sample_data.height, hint)]
    assert [batch.height for batch in batches] == expected_heights

    # Every batch must keep the input column names, order and types
    assert all(batch.schema == sample_data.schema for batch in batches)

    # All batches together must contain the original number of rows
    assert sum(batch.height for batch in batches) == sample_data.height

    # Join the batches and check every value and the row order (counting rows 
    # alone would not detect a missing row replaced by a duplicate)
    assert_frame_equal(combine(batches, sample_data.schema), sample_data)

    # Check the equality with the final result returned by Polars
    assert_frame_equal(actual, sample_data)


@pytest.mark.parametrize("hint", [1, 2, 3, 4])
@pytest.mark.parametrize(
    ("predicate", "failing_hints"),
    [
        pytest.param(pl.col("a") >= 4, [1, 2, 3, 4], id="empty-first"),

        pytest.param(pl.col("a") >= 1, [1, 2, 3, 4], id="partial-first"),

        # First read matches only [0]; later rows match [4, 5]
        pytest.param((pl.col("a") == 0) | (pl.col("a") >= 4), [1, 2, 3, 4], id="partial-and-later"),

        # Matching rows [2, 3] sit on opposite sides of the read boundary
        pytest.param((pl.col("a") >= 2) & (pl.col("a") <= 3), [1, 2, 3], id="boundary-matches"),

        # No rows match. The correct final result is an empty table.
        # This case passed, so it has no expected-failure marker
        pytest.param(pl.col("a") > 100, [], id="no-matches"),
    ],
)
def test_filtered_batches_do_not_stop_at_empty_or_partial_chunk(lmdb_tiny_segment_source, predicate, hint, failing_hints, request):
    """Check that filtered slice sizes do not cause early termination (B2)."""
    # Mark only the hints that reproduced the known batching bug.
    if hint in failing_hints:
        request.node.add_marker(FILTERED_BATCHING)
    
    data = pl.DataFrame({"a": [0, 1, 2, 3, 4, 5]})
    lib = lmdb_tiny_segment_source.library
    lib.write("data", data)

    # Calculate all matching rows independently using only Polars
    expected = oracle(data, lambda frame: frame.filter(predicate))

    # Use the selected hint, which may differ from the stored three-row slices
    with override_batch_hint(hint) as batches:
        # Also record the requests sent to ArcticDB and its returned tables
        with observe_reads() as reads:
            actual = pc.scan_arcticdb(lib, "data", as_of=0).filter(predicate).collect(engine="streaming")

    # Check that at least one executing backend read happened
    assert reads

    # Check that ArcticDB received filtering instructions (this confirms 
    # instructions were sent but it does not check their exact form)
    assert all(read.kwargs["query_builder"] != QueryBuilder() for read in reads)

    # ArcticDB's combined output must already contain the correct filtered rows
    assert_frame_equal(backend_frame(reads), expected)

    # The final result must include every matching row, including later matches
    assert_frame_equal(actual, expected)

    # Check the schema of every emitted batch. If no batches are emitted, 
    # final-result equality above checks the output schema
    assert all(batch.schema == expected.schema for batch in batches)

    # Check that the connector itself delivered the correct filtered rows
    assert_frame_equal(combine(batches, expected.schema), expected)


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_polars_filter_and_limit_orders_are_distinct(lmdb_tiny_segment_source, sample_data, engine):
    """Check that filtering before and after a limit have different meanings."""
    lib = lmdb_tiny_segment_source.library
    lib.write("data", sample_data)

    predicate = pl.col("a") > 1

    # Check the filter before a limit removes any rows
    filtered = sample_data.filter(predicate)
    if not 0 < filtered.height < sample_data.height:
        raise RuntimeError(
            "The test data must contain matching and nonmatching rows"
        )

    # Filter first, then take two matching rows
    expected_first = filtered.head(2)

    # Take two source rows first, then filter those rows
    expected_second = sample_data.head(2).filter(predicate)

    # Execute filtering before the limit through Polarctic
    first = (
        pc.scan_arcticdb(lib, "data", as_of=0)
        .filter(predicate)
        .head(2)
        .collect(engine=engine)
    )

    # Execute the same operations in the opposite order through Polarctic
    second = (
        pc.scan_arcticdb(lib, "data", as_of=0)
        .head(2)
        .filter(predicate)
        .collect(engine=engine)
    )

    # Compare the results correctness
    assert_frame_equal(first, expected_first)
    assert_frame_equal(second, expected_second)


@pytest.mark.parametrize(
    "engine, hint",
    [
        pytest.param("in-memory", None, id="single-read"),
        pytest.param("streaming", 1, id="streaming-small"),
        pytest.param("streaming", 3, id="streaming-aligned"),
        pytest.param("streaming", 4, id="streaming-unaligned"),
    ],
)
@pytest.mark.parametrize("limit", [0, 1, 2, 20])
@pytest.mark.parametrize("fallback", [False, True], ids=["native", "forced-fallback"])
def test_filtered_limit_reaches_late_matching_rows(lmdb_tiny_segment_source, request, engine, hint, limit, fallback):
    """Check that a limit applies to matching rows, not the first source rows."""
    # Put the matching rows at the end of the input to check that the reader gets past the earlier nonmatching rows
    data = pl.DataFrame({"a": [0, 1, 2, 3, 4, 5], "label": ["a", "b", "c", "d", "e", "f"]})
    lib = lmdb_tiny_segment_source.library
    lib.write("late_matches", data)

    # Only a=4 and a=5 match, with labels "e" and "f"
    predicate = pl.col("a") >= 4

    # Check matching rows before the output limit is applied (a zero limit intentionally gives an empty final result)
    filtered = data.filter(predicate)
    if not 0 < filtered.height < data.height:
        raise RuntimeError(
            "The test data must contain matching and nonmatching rows"
        )

    def apply(frame):
        # Filtering comes first: head(1) should therefore return the row with a=4 and label="e"
        # (select label only to also test retaining a until filtering is finished).
        return frame.filter(predicate).head(limit).select("label")

    # Nonzero limits with a batch hint fail when ArcticDB filters the rows (B2 bug).
    # Zero limits, single reads and the Polars fallback pass
    if limit > 0 and hint is not None and not fallback:
        request.node.add_marker(pytest.mark.xfail(
            reason="Known B2 filtered-batching bug",
            strict=True, raises=AssertionError,
        ))

    expected = oracle(data, apply)
    calls = []

    def refuse_translation(expr, query_builder):
        # Record the attempt, then deliberately reject translation 
        # (the connector must still arrange correct filtering using Polars)
        calls.append(expr)
        raise NotImplementedError("forced fallback for late-match limit test")

    # Temporarily use the translator replacement. Restore the translator after each individual test case
    with pytest.MonkeyPatch.context() as mp:
        if fallback:
            # Force translation to fail for the fallback cases, otherwise leave the real translator unchanged
            mp.setattr(pc._TRANSLATOR, "translate", refuse_translation)

        # Give the reader the selected batch hint
        with override_batch_hint(hint):
            query = pc.scan_arcticdb(lib, "late_matches", as_of=0)

            # Run filtering, limiting and selection through Polarctic
            actual = apply(query).collect(engine=engine)

    # With limit=0, Polars may return an empty result without calling the reader.
    # Therefore, do not require a translation attempt for that case
    if fallback and limit > 0:
        # Check that we actually triggered the deliberate translation failure
        if not calls:
            raise RuntimeError(
                "The forced-fallback translator was never called"
            )

    # Check complete values, column names and types
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("batch_size", [None, 2, 3, 20])
@pytest.mark.parametrize("limit", [0, 1, 3, 20])
def test_lazy_source_range_and_polars_limit(lmdb_tiny_segment_source, sample_data, batch_size, limit, request):
    """Preserve a source row range followed by a Polars row limit."""

    # Existing query row ranges interact incorrectly with the reader's
    # additional limit/batch ranges in these reproduced combinations:
    failing_cases = {
        (1, None),
        (1, 2),
        (1, 3),
        (1, 20),
        (3, None),
        (3, 2),
        (3, 3),
        (3, 20),
        (20, 2),
        (20, 3),
    }

    if (limit, batch_size) in failing_cases:
        request.node.add_marker(
            pytest.mark.xfail(
                reason=(
                    "Known range/limit bug: backend read ranges are applied "
                    "before the existing lazy-source row range"
                ),
                strict=True,
                raises=AssertionError,
            )
        )

    lib = lmdb_tiny_segment_source.library
    lib.write("data", sample_data)

    # Create an ArcticDB lazy query
    source = lib.read("data", as_of=0, lazy=True, output_format=OutputFormat.POLARS)

    # Restrict the source to positions 1 through 4
    source = source.row_range((1, 5))

    with override_batch_hint(batch_size):
        actual = (
            pc.scan_arcticdb(source)
            .head(limit)
            .collect(engine="streaming")
        )

    # Calculate the expected range and limit using the original Polars table
    expected = sample_data.slice(1, 4).head(limit)  # head(limit) applies the requested max number of output rows

    assert_frame_equal(actual, expected)


def test_lazy_source_range_beyond_table_is_empty(lmdb_tiny_segment_source, sample_data):
    """Return an empty result for a source range beyond the table."""
    lib = lmdb_tiny_segment_source.library
    lib.write("data", sample_data)

    source = lib.read("data", lazy=True, output_format=OutputFormat.POLARS)

    # The table has 8 rows, so positions 20 to 24 do not exist
    source = source.row_range((20, 25))

    with override_batch_hint(2):
        actual = pc.scan_arcticdb(source).head(4).collect(engine="streaming")

    assert_frame_equal(actual, sample_data.head(0))


@FILTERED_BATCHING
@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_public_scan_reaches_late_matches_with_engine_hint(library_factory, engine):
    """Check late matches without replacing the engine's batch-size hint."""
    lib = library_factory(library_options=LibraryOptions(rows_per_segment=10_000)).library

    # Put the matching rows after a large group of nonmatching rows
    data = pl.DataFrame({"a": pl.Series(range(100_005), dtype=pl.Int64)})
    lib.write("data", data)

    # Only the final five rows match. Early slices can therefore be empty.
    # Do not override the hint, check what Polars actually supplies
    actual = pc.scan_arcticdb(lib, "data", as_of=0).filter(pl.col("a") >= 100_000).collect(engine=engine)

    # Check that the reader reaches and returns those five later rows
    assert_frame_equal(actual, data.tail(5))

    # Polars chooses the batch hint here, so it might request the whole table
    # in one read. The other tests force a small hint to test multiple reads


@FILTERED_BATCHING
def test_streaming_continues_after_full_chunk_then_empty_chunk(lmdb_tiny_segment_source):
    """Check full, empty and matching-again slices in the same scan (B2)."""
    # With a three-row hint, this creates three source groups:
    # [0, 1, 2], [3, 4, 5], and [6, 7, 8]
    data = pl.DataFrame({"a": pl.Series(range(9), dtype=pl.Int64)})
    lib = lmdb_tiny_segment_source.library
    lib.write("middle_empty", data)

    # With three-row slices: keep 0..2, remove 3..5, then keep 6..8
    predicate = (pl.col("a") < 3) | (pl.col("a") >= 6)
    with override_batch_hint(3):
        with observe_reads() as reads:
            actual = pc.scan_arcticdb(lib, "middle_empty", as_of=0).filter(predicate).collect(engine="streaming")

    expected = data.filter(predicate)

    # Check that backend reads happened and received filtering instructions, and that the final result is correct too
    assert reads
    assert all(read.kwargs["query_builder"] != QueryBuilder() for read in reads)
    assert_frame_equal(backend_frame(reads), expected)
    assert_frame_equal(actual, expected)


@pytest.mark.parametrize("engine", [
    "in-memory",
    pytest.param("streaming", id="streaming", marks=FILTERED_BATCHING),   # reproduced the early-stopping bug
])
def test_polars_processing_combines_with_pushdown(lmdb_tiny_segment_source, sample_data, engine):
    """Check filtering, column selection and Polars grouping in one query."""
    lib = lmdb_tiny_segment_source.library
    lib.write("data", sample_data)

    predicate = pl.col("a") >= 1

    # Check the filter before grouping reduces the number of rows
    filtered = sample_data.filter(predicate)
    if not 0 < filtered.height < sample_data.height:
        raise RuntimeError(
            "The test data must contain matching and nonmatching rows"
        )

    def apply(frame):
        # First filter rows, then select grouping inputs, then calculate averages
        return frame.filter(predicate).select("desk", "b").group_by("desk").agg(
            pl.col("b").mean()
        ).sort("desk")

    expected = oracle(sample_data, apply)

    # Use one read for the in-memory case and a small hint for streaming
    hint = None if engine == "in-memory" else 3

    with override_batch_hint(hint), observe_reads() as reads:
        actual = apply(pc.scan_arcticdb(lib, "data")).collect(engine=engine)

    # Collection must execute at least one ArcticDB read
    assert reads

    # Each read must include backend query instructions.
    # This alone does not prove the filter worked; check the returned rows below
    assert all(read.kwargs["query_builder"] != QueryBuilder() for read in reads)

    # The backend must already have removed nonmatching rows
    backend = backend_frame(reads)
    expected_filtered = sample_data.filter(predicate)

    # The columns used by the Polars grouping must be present
    assert {"desk", "b"}.issubset(backend.columns)

    # Compare the grouping inputs in a fixed column order.
    # ArcticDB may also return a because the filter needs it
    assert_frame_equal(backend.select("desk", "b"), expected_filtered.select("desk", "b"))

    # The final grouped result must also be correct
    assert_frame_equal(actual, expected)