"""Group stored row slices into nonoverlapping read ranges for planned ArcticDB reads"""

from collections.abc import Iterator, Sequence


def plan_row_reads(
    slices: Sequence[tuple[int, int]],  # row ranges from ArcticDB's index layout (duplicates removed if same rows are stored across many columns)
    batch_hint: int,  # approximate source rows wanted in each backend read
    row_range: tuple[int | None, int | None] | None = None,  # allowed start/end positions
    n_rows: int | None = None,  # max rows to read from that start (for unfiltered reads only)
) -> Iterator[tuple[int, int]]:
    """Return one (start, end) range for each planned ArcticDB read."""
    if batch_hint <= 0:
        raise ValueError("batch_hint must be positive")

    if n_rows is not None and n_rows < 0:
        raise ValueError("n_rows must be nonnegative")

    # Find which source rows are allowed to be read
    lower, upper = row_range or (None, None)
    lower = 0 if lower is None else lower

    if lower < 0 or (upper is not None and upper < 0):
        raise ValueError("Negative row ranges are not supported by this planner")

    if n_rows is not None:
        # if no filter was pushed, stop after n_rows source rows, starting at lower
        limit = lower + n_rows
        upper = limit if upper is None else min(upper, limit)

    # Build the range for the next backend read
    group_start: int | None = None
    group_end: int | None = None
    previous_end: int | None = None

    for start, end in slices:
        # Every stored slice must contain at least one row
        if start < 0 or end <= start:
            raise ValueError("Invalid stored row slice")

        # Slices must follow each other without gaps or overlaps
        if previous_end is not None and start != previous_end:
            raise ValueError("Stored row slices must be contiguous and ordered")

        previous_end = end

        # Keep only the part inside the requested row range
        start = max(start, lower)
        end = end if upper is None else min(end, upper)

        # Skip slices outside that range
        if start >= end:
            continue

        if group_start is None:
            group_start = start

        # Add this slice to the current read
        group_end = end

        # Include whole slices until we reach or exceed the hint,
        # e.g., a batch_hint of 25 groups three 12-row slices into a 36-row read
        if group_end - group_start >= batch_hint:
            yield group_start, group_end

            # Start a new read range
            group_start = None
            group_end = None

    # Return the remaining slices, even if below the hint
    if group_start is not None:
        assert group_end is not None
        yield group_start, group_end