"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Check version consistency during writes and isolation between query branches."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import polars as pl
import pytest
from arcticdb.version_store.library import Library
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from tests.utils import oracle, override_batch_hint


# Latest-version scans may read a newer version after schema discovery,
# causing inconsistencies between the planned schema and collected data.
#
# Observed failures:
# - AssertionError: collected values differ from the initially resolved version.
# - PanicException: batches have different columns and cannot be joined.
#
# Mark only scans requesting "latest" (as_of=None) as expected failures.
# Scans explicitly requesting version 0 (as_of=0) remain unmarked.
@pytest.mark.parametrize("as_of", [
    pytest.param(
        None, id="latest",
        marks=pytest.mark.xfail(
            reason="Known version bug: latest scan changes version after schema discovery",
            strict=True,
            raises=(AssertionError, pl.exceptions.PanicException),
        ),
    ),
    pytest.param(0, id="version-zero"),
])

# Run each case with the new write happening at one of two points:
# - after the original schema has been discovered;
# - after the first batch read, before the second read
@pytest.mark.parametrize("write_point", ["after_schema", "between_batches"])

# Also test two kinds of new version:
# - changed values but the same columns;
# - changed values plus an extra column.
#
# Both matter: mixed values can be wrong without causing an obvious crash,
# while mixed columns can also make joining batches fail
@pytest.mark.parametrize("change_schema", [False, True], ids=["values-only", "new-column"])
def test_scan_keeps_one_version_while_writer_publishes_new_data(
    arctic_source, sample_data, as_of, write_point, change_schema
):
    """Check version consistency when new data is written before collection or between reads."""
    # Write the original table as version 0
    lib = arctic_source.library
    lib.write("data", sample_data)

    changed = sample_data.with_columns(a=pl.col("a") + 100)
    if change_schema:
        # Add another column in the schema-changing cases
        changed = changed.with_columns(extra=pl.lit("new"))

    # Coordinate the write using signals
    start_write = Event()   # Allow the writer to begin
    write_done = Event()   # Report that the write attempt has finished
    original_read = Library.read
    reads = 0

    def write_new_version():
        # This function runs in the worker thread.
        # Wait for the test to signal the intended write point
        if not start_write.wait(timeout=10):
            raise TimeoutError("The test never requested the write")

        # Write the changed table under the same symbol (create version 1)
        try:
            lib.write("data", changed)

        finally:
            # Signal that the write attempt has ended, even if it raised
            write_done.set()

    def read_and_pause(self, *args, **kwargs):
        # Update the reads counter defined in the outer test
        nonlocal reads

        # Perform the real read before checking whether to trigger the write.
        # In the between-batches case, the first read therefore uses version 0
        result = original_read(self, *args, **kwargs)

        # Count only executing reads through this particular library object
        if self is lib and not kwargs.get("lazy", False):
            reads += 1

            if write_point == "between_batches" and reads == 1:
                # The first batch read has completed so tell the writer to publish version 1 now
                start_write.set()

                # Wait until the write attempt finishes before returning, so the next read happens after the new version exists
                if not write_done.wait(timeout=10):
                    raise TimeoutError("The writer did not finish")

        return result

    # The current reader performs several reads for eight rows with hint = 3
    with override_batch_hint(3):
        query = pc.scan_arcticdb(lib, "data", as_of=as_of)

        # Ask for the schema before starting the writer (at this point only version 0 exists)
        assert query.collect_schema() == sample_data.schema

        # Run the writer in one worker thread while the main thread collects the scan
        with ThreadPoolExecutor(max_workers=1) as pool:
            # Start the writer; it waits until start_write is signalled
            writer = pool.submit(write_new_version)

            try:
                if write_point == "after_schema":
                    # The original schema has already been discovered so write version 1 before collection starts
                    start_write.set()
                    if not write_done.wait(timeout=10):
                        raise TimeoutError("The writer did not finish")

                # Temporarily wrap reads to trigger the write between batches when requested
                with pytest.MonkeyPatch.context() as mp:
                    mp.setattr(Library, "read", read_and_pause)
                    # Collect the original scan. Every batch must still use the version selected when the original schema was discovered
                    actual = query.collect(engine="streaming")

            finally:
                # Release the writer even if collection failed before triggering it
                start_write.set()

                # Wait for completion and propagate any error raised by the writer
                writer.result(timeout=10)

    if write_point == "between_batches":
        # A second read is required to exercise reading after the new write
        if reads < 2:
            raise RuntimeError("This setup did not exercise multiple reads")

    # Check that all values and the schema still match version 0
    assert_frame_equal(actual, sample_data)

    # This checks returned data, not the identity or reuse of storage index keys.
    # Separate instrumentation is needed to verify those storage-level properties.


def test_branches_and_repeated_collection_do_not_share_instructions(arctic_source, sample_data):
    """Check that filters, selections and limits remain separate between queries."""
    # Write one fixed version for every branch to read
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Disable batching to isolate query-state handling from streaming bugs
    with override_batch_hint(None):
        source = pc.scan_arcticdb(lib, "data", as_of=0)

        # Create different queries from the same source.
        # Each branch must keep its own instructions without changing the others

        # Two branches with opposite filters
        lower = source.filter(pl.col("a") < 3)
        upper = source.filter(pl.col("a") >= 3)

        # Return one column and only the first row
        narrow = source.select("a").head(1)

        # Return two columns in a different order and the first four rows
        reordered = source.select("b", "a").head(4)

        # Return three rows starting at position 2, with only the desk column
        middle = source.slice(2, 3).select("desk")

        # Alternate branches with different instructions, 
        # then repeat earlier branches and finally collect the original source.
        #
        # This checks that:
        # - one branch's filter does not affect another;
        # - selections and limits do not leak into other branches;
        # - repeated collection still gives the same answer;
        # - the original source still returns all rows and columns.
        cases = [
            (
                "lower-filter",
                lower,
                lambda frame: frame.filter(pl.col("a") < 3),
            ),
            (
                "one-column-one-row",
                narrow,
                lambda frame: frame.select("a").head(1),
            ),
            (
                "upper-filter",
                upper,
                lambda frame: frame.filter(pl.col("a") >= 3),
            ),
            (
                "reordered-columns",
                reordered,
                lambda frame: frame.select("b", "a").head(4),
            ),
            (
                "middle-rows",
                middle,
                lambda frame: frame.slice(2, 3).select("desk"),
            ),
            (
                "lower-filter-repeated",
                lower,
                lambda frame: frame.filter(pl.col("a") < 3),
            ),
            (
                "narrow-repeated",
                narrow,
                lambda frame: frame.select("a").head(1),
            ),
            (
                "original-source",
                source,
                lambda frame: frame,
            ),
        ]

        for name, query, operation in cases:
            # Calculate this branch's expected result using only Polars
            expected = oracle(sample_data, operation)

            # Apply sanity check for the filtering operations
            if name in {
                "lower-filter",
                "upper-filter",
                "lower-filter-repeated",
            }:
                if not 0 < expected.height < sample_data.height:
                    raise RuntimeError(
                        "The branch filter needs matching and nonmatching rows"
                    )

            # Collect the branch and compare values, order and types
            actual = query.collect()

            try:
                assert_frame_equal(actual, expected)
            except AssertionError as exc:
                # Include the branch name so a failure is easier to locate
                raise AssertionError(
                    f"Incorrect result for branch: {name}"
                ) from exc