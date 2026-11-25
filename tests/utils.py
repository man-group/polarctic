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
import arcticdb.toolbox.query_stats as qs


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
        # Copy the arguments, but keep the original index object.
        # ArcticDB's preloaded index cannot be deep-copied
        recorded = copy.deepcopy(
            {name: value for name, value in kwargs.items() if name != "as_of"}
        )

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
    *,
    record_batches: bool = True,
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
                # Tests can record batches; benchmarks only need the hint override
                if record_batches:
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
def collect_storage_stats() -> Iterator[dict[str, Any]]:
    """Collect fresh ArcticDB query statistics."""
    result: dict[str, Any] = {}

    # Enable recording
    with qs.query_stats():
        qs.reset_stats()

        try:
            # Run the code being measured
            yield result
        finally:
            # Save the counts, even if the code fails
            result.update(qs.get_query_stats())
            qs.reset_stats()

    # Recording stops automatically when the context ends


def query_stats_operation_count(stats, operation, key_type):
    """Return the count for one operation and key type."""
    # Return zero if no count was recorded
    return (
        stats.get("storage_operations", {})
        .get(operation, {})
        .get(key_type, {})
        .get("count", 0)
    )