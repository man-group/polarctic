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
        # Copy arguments before ArcticDB can modify the query builder
        recorded = copy.deepcopy(kwargs)
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