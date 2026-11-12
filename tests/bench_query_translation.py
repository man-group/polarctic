"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source License, use of
this software will be governed by the Apache License, version 2.0.
"""

"""Measure the time to create an ArcticDB QueryBuilder and translate a Polars filter into it."""

import polars as pl
from arcticdb import QueryBuilder

from polarctic.polarctic import PolarsToArcticDBTranslator


def bench_representative_translation(benchmark):
    """Time QueryBuilder creation and Polars-to-ArcticDB filter translation; no data is read."""
    translator = PolarsToArcticDBTranslator()
    expression = (
        (pl.col("a") > 0)
        & (pl.col("b") < 100)
        & ((pl.col("c") == 5) | (pl.col("c") == 10))
    )

    def translate():
        # Start with a new empty ArcticDB query builder each time to avoid accumulating filters
        return translator.translate(expression, QueryBuilder())

    # Construct the ArcticDB query before the timed function starts
    expected = QueryBuilder()
    expected = expected[
        (expected["a"] > 0)
        & (expected["b"] < 100)
        & ((expected["c"] == 5) | (expected["c"] == 10))
    ]

    # Check that translation constructs the intended query before timing it
    assert translate() == expected

    # Pytest-benchmark will call translate repeatedly and measure its duration.
    # Translation has already run once, so repeated calls may use the parser cache.
    # These timings therefore represent repeated use, not first-use translation
    benchmark(translate)