"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.
"""

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest
from arcticdb.version_store.library import Library


@dataclass
class ObservedRead:
    """Read arguments and the Polars dataframe returned by ArcticDB."""

    kwargs: dict[str, Any]
    data: pl.DataFrame


@contextmanager
def observe_reads() -> Iterator[list[ObservedRead]]:
    """Record executing Library.read calls made inside the with block."""
    reads: list[ObservedRead] = []
    original = Library.read

    def observe(self: Library, *args: Any, **kwargs: Any) -> Any:
        # Copy options such as QueryBuilder before ArcticDB can modify them.
        # Do not deepcopy the preloaded index: it is a native object that cannot
        # be pickled, and preserving its reference is appropriate for observation
        recorded = copy.deepcopy(
            {
                name: value
                for name, value in kwargs.items()
                if name != "as_of"
            }
        )

        # Retain the original version selector or preloaded index object.
        if "as_of" in kwargs:
            recorded["as_of"] = kwargs["as_of"]

        result = original(self, *args, **kwargs)

        # Lazy calls prepare a read; they do not return the table yet
        if not kwargs.get("lazy", False):
            assert isinstance(result.data, pl.DataFrame)
            reads.append(ObservedRead(recorded, result.data))

        return result

    # Restore Library.read automatically when the block ends
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Library, "read", observe)
        yield reads


@contextmanager
def override_batch_hint(
    size: int | None,
) -> Iterator[list[pl.DataFrame]]:
    """Override the hint and record batches for scans created in this block."""
    batches: list[pl.DataFrame] = []
    original = pl.io.plugins.register_io_source

    def register(
        io_source: Callable[..., Iterator[pl.DataFrame]],
        **kwargs: Any,
    ) -> pl.LazyFrame:
        def source(
            columns: Any,
            predicate: Any,
            n_rows: Any,
            supplied_size: Any,
        ) -> Iterator[pl.DataFrame]:
            # Preserve all arguments except the batch-size hint
            for batch in io_source(columns, predicate, n_rows, size):
                batches.append(batch)
                yield batch

        return original(io_source=source, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pl.io.plugins, "register_io_source", register)
        yield batches


def oracle(
    data: pl.DataFrame,
    operation: Callable[[pl.LazyFrame], pl.LazyFrame],
) -> pl.DataFrame:
    """Apply the operation to known input using only Polars."""
    return operation(data.lazy()).collect(engine="in-memory")


def backend_frame(reads: list[ObservedRead]) -> pl.DataFrame:
    """Combine data returned directly by ArcticDB."""
    assert reads, "No executing ArcticDB read was observed"
    return pl.concat([read.data for read in reads])



@contextmanager
def collect_storage_stats():
    """Collect fresh standard and enterprise stats inside the block."""
    from arcticdb.toolbox import query_stats as qs
    from arcticdb_ext.tools import query_stats as core_qs
    from arcticdb_enterprise_ext import query_stats as enterprise_qs

    # Use both existing trackers and their result functions
    trackers = [
        ("core", qs, qs.get_query_stats),
        ("enterprise", enterprise_qs, enterprise_qs.get_stats),
    ]

    # Do not reset or disable another collector's measurements
    if core_qs.is_enabled() or enterprise_qs.is_enabled():
        raise RuntimeError("Another query-stats collector is already enabled")

    result = {}
    enabled = []

    try:
        # Clear old measurements and start recording
        for _, tracker, _ in trackers:
            tracker.reset_stats()
            tracker.enable()
            enabled.append(tracker)

        yield result

    finally:
        # Stop recording, including when the block raises an error
        for tracker in reversed(enabled):
            tracker.disable()

        # Keep the two trackers' results separate
        for name, _, getter in trackers:
            result[name] = getter()


def storage_read_count(stats, key_type):
    """Add the S3 read counts for a key type from both trackers."""
    total = 0

    for snapshot in stats.values():
        # Find S3 read measurements in this tracker's results
        operations = snapshot.get("storage_operations", {})
        reads = operations.get("S3_GetObject", {})

        # We need counts grouped by key type, not one overall count
        if "count" in reads:
            raise AssertionError(
                "This query-stats version lacks per-key-type S3 counters"
            )

        # Missing entries contribute zero
        total += reads.get(key_type, {}).get("count", 0)

    return total