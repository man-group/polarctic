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
    """Measure the time to translate a large filter into an ArcticDB query."""
    translator = PolarsToArcticDBTranslator()

    # Create 256 small Polars filters:
    # a > 0, a > 1, a > 2, ..., a > 255.
    #
    # Each comparison contains several expression nodes: a column, a number and a comparison operation.
    # Combining them creates a filter with more than 1000 nodes
    expressions = [pl.col("a") > value for value in range(256)]

    # Build the same small filters directly in ArcticDB (independent reference for checking the translator)
    expected = QueryBuilder()
    backend_expressions = [expected["a"] > value for value in range(256)]

    # Join the filters using AND. Join them in pairs so the expression tree remains shallow
    while len(expressions) > 1:
        expressions = [
            expressions[index] & expressions[index + 1]
            for index in range(0, len(expressions), 2)
        ]

        # Combine the matching ArcticDB filters in exactly the same order
        backend_expressions = [
            backend_expressions[index] & backend_expressions[index + 1]
            for index in range(0, len(backend_expressions), 2)
        ]

    # Only one combined Polars filter remains at the end
    expression = expressions[0]

    # Add the combined reference filter to the ArcticDB query builder
    expected = expected[backend_expressions[0]]

    def translate():
        """Translate the large Polars filter into a new ArcticDB query."""
        # Start with an empty builder on every call
        return translator.translate(expression, QueryBuilder())

    # Check correctness before measuring speed
    assert translate() == expected

    # Time repeated calls to translate().
    # The correctness check already translated this expression once, so its parsed expression may be cached
    # Therefore, it measures repeated use, not the first translation of a new expression
    benchmark(translate)


    # Save these labels with the benchmark results
    benchmark.extra_info["comparison_count"] = 256
    benchmark.extra_info["measurement"] = "warm repeated translation"