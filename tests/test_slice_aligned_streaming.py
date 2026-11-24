import polars as pl
import pytest
import datetime as dt
from arcticdb import Arctic, LibraryOptions, OutputFormat
from polars.testing import assert_frame_equal

import polarctic.polarctic as pc
from polarctic.streaming import plan_row_reads
from tests.utils import collect_storage_stats, enable_polars_writes, observe_reads, override_batch_hint, query_stats_operation_count


@pytest.mark.parametrize(('hint', 'expected'), [
    (25, [(0, 36), (36, 48)]),
    (24, [(0, 24), (24, 48)]),
    (1, [(0, 12), (12, 24), (24, 36), (36, 48)]),
    (100, [(0, 48)]),
])
def test_greedy_groups(hint, expected):
    assert list(plan_row_reads([(0, 12), (12, 24), (24, 36), (36, 48)], hint)) == expected


def test_partial_range_and_limit():
    slices = [(0, 12), (12, 24), (24, 36), (36, 48)]
    assert list(plan_row_reads(slices, 25, (5, 40))) == [(5, 36), (36, 40)]
    assert list(plan_row_reads(slices, 25, (5, 40), 10)) == [(5, 15)]
    assert list(plan_row_reads(slices, 25, n_rows=0)) == []
    assert list(plan_row_reads([], 25)) == []


@pytest.mark.parametrize('hint', [0, -1])
def test_invalid_hint(hint):
    with pytest.raises(ValueError):
        list(plan_row_reads([(0, 12)], hint))


@pytest.fixture
def slice_library():
    ac = Arctic('mem://', output_format=OutputFormat.POLARS)
    lib = ac.create_library('slice_tests', library_options=LibraryOptions(rows_per_segment=12, columns_per_segment=2))
    enable_polars_writes(lib)
    try:
        yield lib
    finally:
        ac.delete_library('slice_tests')


@pytest.mark.parametrize('hint', [1, 24, 25, 100])
@pytest.mark.parametrize('projection', [False, True])
def test_each_required_segment_fetched_once(slice_library, hint, projection):
    lib = slice_library
    # Four row slices, each containing two column segments.
    data = pl.DataFrame({name: list(range(48)) for name in ['a', 'b', 'c', 'd']})
    lib.write('data', data)
    columns = ['a'] if projection else None
    expected = data.select('a') if projection else data
    # Check the counters against a normal direct read first
    with collect_storage_stats() as baseline_stats:
        baseline = lib.read('data', as_of=0, columns=columns).data
    assert_frame_equal(baseline, expected)
    baseline_count = query_stats_operation_count(baseline_stats, 'Memory_GetObject', 'TABLE_DATA')
    assert baseline_count == (4 if projection else 8)

    with override_batch_hint(hint):
        query = pc.scan_arcticdb(lib, 'data', as_of=0)
        if projection:
            query = query.select('a')
        # Keep schema and layout reads outside collection measurement
        query.collect_schema()
        with observe_reads() as reads, collect_storage_stats() as stats:
            actual = query.collect(engine='streaming')
    assert_frame_equal(actual, expected)
    assert [read.kwargs['row_range'] for read in reads] == list(
        plan_row_reads([(0, 12), (12, 24), (24, 36), (36, 48)], hint)
    )
    assert query_stats_operation_count(stats, 'Memory_GetObject', 'TABLE_INDEX') == 0
    assert query_stats_operation_count(stats, 'Memory_GetObject', 'TABLE_DATA') == baseline_count


def test_layout_keeps_resolved_version(slice_library):
    lib = slice_library
    data = pl.DataFrame({'a': list(range(48))})
    lib.write('data', data)
    with override_batch_hint(25):
        query = pc.scan_arcticdb(lib, 'data')
        query.collect_schema()
        # Publishing a newer version must not change this scan's layout/data
        lib.append('data', pl.DataFrame({'a': [48, 49]}))
        actual = query.collect(engine='streaming')
    assert_frame_equal(actual, data)


def test_unequal_slices_and_limit(slice_library):
    lib = slice_library
    lib.write('data', pl.DataFrame({'a': list(range(5))}))
    lib.append('data', pl.DataFrame({'a': list(range(5, 22))}))
    # Stored slices: [0,5), [5,17), [17,22)
    with override_batch_hint(10):
        query = pc.scan_arcticdb(lib, 'data', as_of=1).head(19)
        with observe_reads() as reads:
            actual = query.collect(engine='streaming')
    assert_frame_equal(actual, pl.DataFrame({'a': list(range(19))}))
    assert [read.kwargs['row_range'] for read in reads] == [(0, 17), (17, 19)]


@pytest.mark.parametrize('hint', [1, 25])
@pytest.mark.parametrize('pre_applied', [False, True])
def test_filter_continues_after_empty_ranges(slice_library, hint, pre_applied):
    lib = slice_library
    data = pl.DataFrame({'a': list(range(48)), 'b': list(range(48))})
    lib.write('data', data)
    with override_batch_hint(hint):
        if pre_applied:
            source = lib.read('data', as_of=0, lazy=True, output_format=OutputFormat.POLARS)
            source = source[source['a'] >= 36]
            query = pc.scan_arcticdb(source).filter(pl.col('a') < 45)
        else:
            query = pc.scan_arcticdb(lib, 'data', as_of=0).filter(
                (pl.col('a') >= 36) & (pl.col('a') < 45)
            )
        query.collect_schema()
        with observe_reads() as reads, collect_storage_stats() as stats:
            actual = query.collect(engine='streaming')
    expected = data.filter((pl.col('a') >= 36) & (pl.col('a') < 45))
    assert_frame_equal(actual, expected)
    # The empty early ranges must not stop subsequent reads
    assert [read.kwargs['row_range'] for read in reads] == list(
        plan_row_reads([(0, 12), (12, 24), (24, 36), (36, 48)], hint)
    )
    assert query_stats_operation_count(stats, 'Memory_GetObject', 'TABLE_INDEX') == 0
    assert query_stats_operation_count(stats, 'Memory_GetObject', 'TABLE_DATA') == 4


@pytest.mark.parametrize("timeseries", [False, True], ids=["row-index", "time-index"])
@pytest.mark.parametrize("filtered", [False, True], ids=["plain", "filtered"])
def test_preloaded_index_uses_aligned_reads_without_planning_data_fetches(
    slice_library, timeseries, filtered
):
    """Keep an already-loaded version and plan reads without fetching its data."""
    lib = slice_library

    # The fixture uses 12-row slices, so with 48 rows it should create four stored row slices
    data = pl.DataFrame({"a": list(range(48))})

    if timeseries:
        # Also check timestamp-indexed tables.
        # columns=[] alone can still read timestamp data, so this case
        # checks that the zero-row version probe avoids those fetches
        timestamps = pl.datetime_range(
            dt.datetime(2020, 1, 1),
            dt.datetime(2020, 2, 17),
            interval="1d",
            time_unit="ns",
            eager=True,
        )
        data = data.with_columns(timestamps.alias("ts")).select("ts", "a")

    lib.write("data", data, index_column=timeseries)

    # Load version 0's index before passing the source to Polarctic
    source = lib.read(
        "data",
        lazy=True,
        output_format=OutputFormat.POLARS,
    )
    source._collect_schema()
    original_index = source._preloaded_index

    # Publish different values as version 1. 
    # The scan must still use the index already loaded for version 0
    newer_data = data.with_columns(a=pl.col("a") + 100)
    lib.write("data", newer_data, index_column=timeseries)

    # Confirm the new version really exists, outside our IO measurements
    latest = lib.read("data", output_format=OutputFormat.POLARS)
    assert latest.version == 1
    assert_frame_equal(latest.data, newer_data)

    expected = data.filter(pl.col("a") >= 36) if filtered else data

    with override_batch_hint(25):
        # Confirm that the empty-row version probe and layout loading do not fetch table-data keys
        with collect_storage_stats() as planning:
            query = pc.scan_arcticdb(source)

            if filtered:
                # The first planned range contains no matching rows but the reader must still reach the final range
                query = query.filter(pl.col("a") >= 36)

            query.collect_schema()

        assert query_stats_operation_count(
            planning, "Memory_GetObject", "TABLE_DATA"
        ) == 0

        # Now measure only the actual data reads
        with observe_reads() as reads, collect_storage_stats() as collection:
            actual = query.collect(engine="streaming")

    # Return version 0's values, not version 1's replacement values
    assert_frame_equal(actual, expected)

    # The hint of 25 groups three 12-row slices, then the final slice
    assert [read.kwargs["row_range"] for read in reads] == [
        (0, 36),
        (36, 48),
    ]

    # Data reads must use the original loaded index
    assert all(
        read.kwargs["as_of"] is original_index
        for read in reads
    )

    # Collection must not fetch the index again
    assert query_stats_operation_count(
        collection, "Memory_GetObject", "TABLE_INDEX"
    ) == 0

    # This table has one column segment per row slice.
    # Both plain and filtered reads should fetch four data segments here
    assert query_stats_operation_count(
        collection, "Memory_GetObject", "TABLE_DATA"
    ) == 4

    # Polarctic must not change the original query's version setting or index
    assert source.read_request.as_of is None
    assert source._preloaded_index is original_index
