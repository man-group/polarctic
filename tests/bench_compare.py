"""
Copyright 2026 Man Group Operations Limited

Use of this software is governed by the Business Source License 1.1 included in the file LICENSE

As of the Change Date specified in that file, in accordance with the Business Source
License, use of this software will be governed by the Apache License, version 2.0.

Compares polarctic scan_arcticdb() against direct ArcticDB lib.read() and prints a
speedup/overhead ratio table.

Benchmarking policy:
- One-time setup (LMDB init, LazyFrame construction, QB construction) is excluded.
- Each scenario is validated once before timing to ensure both implementations
    return the same result.
- Timed sections use pyperf worker processes and disable GC during each sample
    to reduce run-to-run noise.

Usage:
        uv run --extra dev python tests/bench_compare.py
        uv run --extra dev python tests/bench_compare.py --fast
        uv run --extra dev python tests/bench_compare.py --rigorous -o benchmark-results.json
"""

import gc
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pandas.testing as pdt
import polars as pl
import pyperf
from arcticdb import Arctic, OutputFormat, QueryBuilder
from arcticdb.version_store.library import Library

import polarctic.polarctic as polarctic_module

_SIMPLE_FILTER_EXPR = pl.col("a") > 500
_COMPOUND_FILTER_EXPR = (pl.col("a") > 200) & (pl.col("b") < 700.0)
_TWO_COLUMN_PROJECTION = ["a", "b"]
_NOISE_WARNING_THRESHOLD = 10.0


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    polarctic_fn: Callable[[], object]
    baseline_fn: Callable[[], object]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_pandas(result: object) -> pd.DataFrame:
    if isinstance(result, pl.DataFrame):
        return result.to_pandas()
    if isinstance(result, pd.DataFrame):
        return result
    raise TypeError(f"Unsupported benchmark result type: {type(result)!r}")


def _verify_case(case: BenchmarkCase) -> None:
    polarctic_df = _result_to_pandas(case.polarctic_fn())
    baseline_df = _result_to_pandas(case.baseline_fn())
    pdt.assert_frame_equal(
        polarctic_df,
        baseline_df,
        check_dtype=False,
        check_like=True,
        obj=case.label,
    )


def _time_callable(loops: int, fn: Callable[[], object]) -> float:
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        timer = time.perf_counter
        start = timer()
        for _ in range(loops):
            fn()
        return timer() - start
    finally:
        if gc_was_enabled:
            gc.enable()


def _qb_simple() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[qb["a"] > 500]
    return cast(QueryBuilder, qb)


def _qb_compound() -> QueryBuilder:
    qb: Any = QueryBuilder()
    qb = qb[(qb["a"] > 200) & (qb["b"] < 700.0)]
    return cast(QueryBuilder, qb)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _setup(tmp_dir: Path) -> dict[str, Library]:
    lmdb_dir = tmp_dir / "lmdb"
    lmdb_dir.mkdir(parents=True)
    uri = f"lmdb://{lmdb_dir}"

    ac = Arctic(uri)
    lib: Library = ac.create_library("bench")

    rng = np.random.default_rng(42)

    def _df(n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "a": rng.integers(0, 1000, size=n).astype(np.int64),
                "b": rng.uniform(0.0, 1000.0, size=n).astype(np.float64),
                "label": [f"item_{i % 50}" for i in range(n)],
                "ts": pd.date_range("2020-01-01", periods=n, freq="s"),
            }
        )

    lib.write("medium", _df(10_000))
    lib.write("large", _df(100_000))

    return {"lib": lib}


# ---------------------------------------------------------------------------
# Benchmark pairs
# ---------------------------------------------------------------------------


def _build_cases(lib: Library) -> list[BenchmarkCase]:
    """Return the comparison scenarios for polarctic and raw ArcticDB."""
    qb_simple = _qb_simple()
    qb_compound = _qb_compound()

    lazy_medium = polarctic_module.scan_arcticdb(lib, "medium")
    lazy_large = polarctic_module.scan_arcticdb(lib, "large")

    lazy_filter_simple_medium = lazy_medium.filter(_SIMPLE_FILTER_EXPR)
    lazy_filter_simple_large = lazy_large.filter(_SIMPLE_FILTER_EXPR)
    lazy_filter_compound_medium = lazy_medium.filter(_COMPOUND_FILTER_EXPR)
    lazy_filter_compound_large = lazy_large.filter(_COMPOUND_FILTER_EXPR)

    lazy_select_two_columns_medium = lazy_medium.select(*_TWO_COLUMN_PROJECTION)
    lazy_select_two_columns_large = lazy_large.select(*_TWO_COLUMN_PROJECTION)

    return [
        BenchmarkCase(
            "Full scan - medium (10k rows)",
            lazy_medium.collect,
            lambda: lib.read("medium", output_format=OutputFormat.PANDAS).data,
        ),
        BenchmarkCase(
            "Full scan - large (100k rows)",
            lazy_large.collect,
            lambda: lib.read("large", output_format=OutputFormat.PANDAS).data,
        ),
        BenchmarkCase(
            "Filter simple (a > 500) - medium",
            lazy_filter_simple_medium.collect,
            lambda: lib.read(
                "medium",
                query_builder=qb_simple,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        BenchmarkCase(
            "Filter simple (a > 500) - large",
            lazy_filter_simple_large.collect,
            lambda: lib.read(
                "large",
                query_builder=qb_simple,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        BenchmarkCase(
            "Filter compound (a>200 & b<700) - medium",
            lazy_filter_compound_medium.collect,
            lambda: lib.read(
                "medium",
                query_builder=qb_compound,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        BenchmarkCase(
            "Filter compound (a>200 & b<700) - large",
            lazy_filter_compound_large.collect,
            lambda: lib.read(
                "large",
                query_builder=qb_compound,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        BenchmarkCase(
            "Select 2 columns - medium",
            lazy_select_two_columns_medium.collect,
            lambda: lib.read(
                "medium",
                columns=_TWO_COLUMN_PROJECTION,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
        BenchmarkCase(
            "Select 2 columns - large",
            lazy_select_two_columns_large.collect,
            lambda: lib.read(
                "large",
                columns=_TWO_COLUMN_PROJECTION,
                output_format=OutputFormat.PANDAS,
            ).data,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_COL_LABEL = 42
_COL_NUM = 10


def _benchmark_name(label: str, implementation: str) -> str:
    return f"{label} [{implementation}]"


def _median_ms(benchmark: pyperf.Benchmark) -> float:
    return float(benchmark.median()) * 1_000


def _coefficient_of_variation(benchmark: pyperf.Benchmark) -> float:
    mean = float(benchmark.mean())
    if mean == 0:
        return 0.0
    return float(benchmark.stdev()) / mean * 100


def _row(label: str, polar_ms: float, base_ms: float) -> str:
    ratio = base_ms / polar_ms
    overhead_ms = polar_ms - base_ms
    sign = "+" if overhead_ms >= 0 else "-"
    return (
        f"  {label:<{_COL_LABEL}}"
        f"  {polar_ms:{_COL_NUM}.2f} ms"
        f"  {base_ms:{_COL_NUM}.2f} ms"
        f"  {ratio:{_COL_NUM}.2f}x"
        f"  {sign}{abs(overhead_ms):.2f} ms"
    )


def _print_results(results: list[tuple[str, pyperf.Benchmark, pyperf.Benchmark]]) -> None:
    header_label = "Scenario"
    header = (
        f"\n  {header_label:<{_COL_LABEL}}"
        f"  {'polarctic':>{_COL_NUM + 3}}"
        f"  {'arcticdb':>{_COL_NUM + 3}}"
        f"  {'ratio':>{_COL_NUM + 1}}"
        f"  overhead"
    )
    sep = "  " + "-" * (_COL_LABEL + 2 * (_COL_NUM + 5) + (_COL_NUM + 3) + 12)

    print("\nResults (pyperf median wall-clock time)")
    print(sep)
    print(header)
    print(sep)

    noisy_cases: list[tuple[str, float, float]] = []
    for label, polar_benchmark, baseline_benchmark in results:
        polar_ms = _median_ms(polar_benchmark)
        baseline_ms = _median_ms(baseline_benchmark)
        print(_row(label, polar_ms, baseline_ms))

        polar_noise = _coefficient_of_variation(polar_benchmark)
        baseline_noise = _coefficient_of_variation(baseline_benchmark)
        if max(polar_noise, baseline_noise) >= _NOISE_WARNING_THRESHOLD:
            noisy_cases.append((label, polar_noise, baseline_noise))

    print(sep)
    print(
        "  ratio > 1x -> polarctic faster than raw ArcticDB  |"
        "  ratio < 1x -> overhead (positive overhead column)"
    )

    if noisy_cases:
        print(
            "\nPotentially noisy scenarios "
            f"(coefficient of variation >= {_NOISE_WARNING_THRESHOLD:.0f}%):"
        )
        for label, polar_noise, baseline_noise in noisy_cases:
            print(
                f"  {label}: polarctic={polar_noise:.1f}%  arcticdb={baseline_noise:.1f}%"
            )
        print(
            "  rerun with --rigorous or pin the benchmark to a CPU with --affinity"
            " if these scenarios matter"
        )


def main() -> None:
    runner = pyperf.Runner(
        values=4,
        processes=4,
        min_time=0.2,
        warmups=2,
        metadata={"benchmark_suite": "polarctic comparison"},
    )
    args = runner.parse_args()

    with tempfile.TemporaryDirectory(prefix="polarctic_bench_") as tmp:
        store = _setup(Path(tmp))
        lib = store["lib"]
        cases = _build_cases(lib)

        if not args.worker and not args.quiet:
            print("Verifying benchmark outputs ...")
        if not args.worker:
            for case in cases:
                if not args.quiet:
                    print(f"  checking: {case.label}")
                _verify_case(case)

        results: list[tuple[str, pyperf.Benchmark | None, pyperf.Benchmark | None]] = []
        for case in cases:
            polar_benchmark = runner.bench_time_func(
                _benchmark_name(case.label, "polarctic"),
                _time_callable,
                case.polarctic_fn,
                metadata={"implementation": "polarctic", "scenario": case.label},
            )
            baseline_benchmark = runner.bench_time_func(
                _benchmark_name(case.label, "arcticdb"),
                _time_callable,
                case.baseline_fn,
                metadata={"implementation": "arcticdb", "scenario": case.label},
            )
            results.append((case.label, polar_benchmark, baseline_benchmark))

        if args.worker or args.compare_to:
            return

        completed_results = [
            (label, polar_benchmark, baseline_benchmark)
            for label, polar_benchmark, baseline_benchmark in results
            if polar_benchmark is not None and baseline_benchmark is not None
        ]
        _print_results(completed_results)


if __name__ == "__main__":
    main()
