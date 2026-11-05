# Polarctic tests: first review batch and next steps

## Purpose
The RFC review identified two main problems with the existing tests: correct final output does not prove that ArcticDB performed the filtering, and 
constructing a QueryBuilder does not prove that the expression executes correctly on stored data.
This first batch adds real-storage execution checks, direct Polars comparisons, and observations of the requests and Arrow tables returned by ArcticDB.

## First-batch scope
The intended repository files are:
`tests/conftest.py` : Real temporary LMDB storage, known input data, URI/library scan factory, read observer, batch-hint control, and independent Polars reference calculation
`tests/test_scan_arcticdb.py` : Entry points, schema, reads, filters, selection, calculations, grouping, explicit versions, argument validation, and Arrow output normalisation
`tests/test_pushdown.py` : Backend filter/projection requests and results; filter-only column dependencies; calculations/grouping remaining in Polars; planning versus collection
`tests/test_query_translation.py` : Query construction and selected current text-parser helper behaviour
`tests/test_predicate_execution.py` : Expression correctness, selected native-pushdown cases, forced translation failure, real difficult expressions, and operation-order combinations

## Results to date
First-batch test run: 116 passed, 38 failed. Failures expose missing/incorrect unsupported-filter handling, ignored literal string options, rejected integer 
bitwise queries, incorrect filtered-limit results, and translator error-handling gaps.

The scan and pushdown tests, including the selected basic native-predicate cases, had no failures in this run.

## Remaining test areas
The following areas are planned for later test batches.

### 1. Streaming and output limits — `tests/test_streaming.py`
Check that batched reads return all matching rows, including matches after empty or partially matching batches. Verify row order, batch schemas, and correct operation order when combining 
filtering, selection and limits. Test both native ArcticDB filtering and Polars fallback.

### 2. Storage-slice efficiency — `tests/test_storage_io.py`
Check that small batch hints do not cause unnecessary repeated reads or decoding of stored segments. Verify that column selection avoids reading unrelated column segments.

### 3. Schema and index reuse — `tests/test_schema_io.py`
Check that query planning and schema discovery do not read table-data segments. Verify that collection reuses the loaded index and that all reads within a scan use the same resolved index.

### 4. Concurrent writes and independent queries — `tests/test_version_consistency.py`
Check that a scan stays on one dataset version while new versions are written. Verify that repeated collection and separate queries from the same source do not share or accumulate filters, selections or limits.

### 5. Conversion and failures — `tests/test_conversion_and_failures.py`
Check values and types for numeric, Boolean, string and timestamp data, including nulls, sparse columns and empty results. Verify useful errors for unsupported data and ensure read or conversion failures cannot silently return partial results.

### 6. Timeseries-index filtering — shared fixtures and pushdown/storage tests
Use a genuine datetime-indexed dataset to check time-range filtering, boundary behaviour and skipping of out-of-range storage slices. Combine time conditions with unsupported filters without losing either condition.
The current timestamp-column tests do not cover this storage behaviour.

### 7. Performance comparisons — existing benchmark files
Compare Polarctic pushdown with an ArcticDB Arrow read followed by equivalent processing in Polars. Cover time filters, column selection and streaming, using the same fixed dataset version. 
Check results before timing and distinguish fresh-plan from reused-plan measurements.
Retain one representative translation benchmark and measure statistics overhead where supported.

### 8. Deferred translator redesign and compatibility
Replacing `str(expr)` parsing with the Polars expression tree is not implemented yet.
When implemented, replace obsolete text-parser helper tests with real-expression tests for unsupported shapes/options, quoted strings, backslashes, parser-like literal text, unusual column names, and the supported Polars versions.

Partial AND pushdown is optional future work. Correct whole-expression fallback has been agreed to be sufficient for the MVP. For an OR filter, applying only one side in ArcticDB could wrongly remove rows that match the other side.

As passing an ArcticDB LazyDataFrame to Polarctic has been agreed to be excluded from the MVP, ArcticDB preprocessing/resampling combinations are deferred.
Polars grouping tests do not replace ArcticDB preprocessing coverage.