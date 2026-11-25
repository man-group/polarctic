"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

"""Check version consistency during writes and isolation between query branches."""

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
    arctic_source, sample_data, monkeypatch, as_of, write_point, change_schema
):
    """Keep the original version when a write happens before or between reads."""
    # Write the original table as version 0
    lib = arctic_source.library
    lib.write("data", sample_data)

    changed = sample_data.with_columns(a=pl.col("a") + 100)
    if change_schema:
        # Add another column in the schema-changing cases
        changed = changed.with_columns(extra=pl.lit("new"))

    original_read = Library.read
    executing_reads = 0

    def read_and_write(self, *args, **kwargs):
        """Write version 1 after the first executing read finishes."""
        nonlocal executing_reads

        # Complete the real read before publishing the new version
        result = original_read(self, *args, **kwargs)

        # Ignore lazy setup and reads from other libraries
        if self is lib and not kwargs.get("lazy", False):
            executing_reads += 1

            if write_point == "between_batches" and executing_reads == 1:
                # Publish version 1 before the reader requests its next batch
                lib.write("data", changed)

        return result

    # The current reader performs several reads for eight rows with hint = 3
    with override_batch_hint(3):
        query = pc.scan_arcticdb(lib, "data", as_of=as_of)

        # Discover the schema while only version 0 exists
        assert query.collect_schema() == sample_data.schema

        if write_point == "after_schema":
            # Publish version 1 before any data is collected
            lib.write("data", changed)

        # Wrap reads only after schema discovery
        with monkeypatch.context() as mp:
            mp.setattr(Library, "read", read_and_write)
            actual = query.collect(engine="streaming")

    if write_point == "between_batches":
        # Without a second read, we would not test reading after the write
        assert executing_reads >= 2, ("The test must perform another read after publishing version 1")

    # Every returned row and column must still match version 0
    assert_frame_equal(actual, sample_data)



@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda frame: frame.filter(pl.col("a") < 3),
            id="lower-filter",
        ),
        pytest.param(
            lambda frame: frame.filter(pl.col("a") >= 3),
            id="upper-filter",
        ),
        pytest.param(
            lambda frame: frame.select("a").head(1),
            id="one-column-one-row",
        ),
        pytest.param(
            lambda frame: frame.select("b", "a").head(4),
            id="reordered-columns",
        ),
        pytest.param(
            lambda frame: frame.slice(2, 3).select("desk"),
            id="middle-rows",
        ),
        pytest.param(
            lambda frame: frame,
            id="original-source",
        ),
    ],
)
def test_repeated_collection_preserves_query(arctic_source, sample_data, operation):
    """Collect each operation twice without changing its result."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    # Calculate the expected result using only the original Polars table
    expected = oracle(sample_data, operation)

    # Disable batching so this test focuses on query state
    with override_batch_hint(None):
        source = pc.scan_arcticdb(lib, "data")
        query = operation(source)

        first_collect = query.collect()
        second_collect = query.collect()

    # Both collections must return the same correct values and types
    assert_frame_equal(first_collect, expected)
    assert_frame_equal(second_collect, expected)


def test_branches_do_not_change_each_other(arctic_source, sample_data):
    """Keep different branches and their shared source independent."""
    lib = arctic_source.library
    lib.write("data", sample_data)

    with override_batch_hint(None):
        # Build several branches from the same scan
        source = pc.scan_arcticdb(lib, "data")
        lower = source.filter(pl.col("a") < 3)
        narrow = source.select("a").head(1)
        upper = source.filter(pl.col("a") >= 3)

        # Alternate between branches to catch instructions leaking between them
        assert_frame_equal(lower.collect(), sample_data.filter(pl.col("a") < 3))
        assert_frame_equal(narrow.collect(), sample_data.select("a").head(1))
        assert_frame_equal(upper.collect(), sample_data.filter(pl.col("a") >= 3))

        # Collect an earlier branch again after using the other branches
        assert_frame_equal(lower.collect(), sample_data.filter(pl.col("a") < 3))

        # The shared source must still return every original row and column
        assert_frame_equal(source.collect(), sample_data)