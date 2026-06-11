-- F1 predictor — DuckDB views over the pipeline's Parquet artefacts.
-- Run this once in your DuckDB client (connected to any .duckdb file) to expose
-- the parquet outputs as browsable views. Paths are absolute so it works from
-- any working directory. Re-run safely: CREATE OR REPLACE.
--
-- Notes:
--   * snapshots.position is STANDARDISED (not 1-20); final_position/relevance are raw.
--   * raw intervals.gap_to_leader is a STRING ("+1 LAP"); use TRY_CAST(... AS DOUBLE).
--   * car_data is large (~700k rows/race) and may be mid-backfill — filter/aggregate.

-- Stage 4 snapshots (training tensors)
CREATE OR REPLACE VIEW snap_train AS
  SELECT * FROM '/mnt/c/users/neils/projects/python-projects/f1-finishing-positions/.claude/worktrees/pipeline-foundation/data/snapshots/train.parquet';
CREATE OR REPLACE VIEW snap_val AS
  SELECT * FROM '/mnt/c/users/neils/projects/python-projects/f1-finishing-positions/.claude/worktrees/pipeline-foundation/data/snapshots/val.parquet';
CREATE OR REPLACE VIEW snap_test AS
  SELECT * FROM '/mnt/c/users/neils/projects/python-projects/f1-finishing-positions/.claude/worktrees/pipeline-foundation/data/snapshots/test.parquet';

-- Stage 3 features (one file per race; filename column carries the path)
CREATE OR REPLACE VIEW features AS
  SELECT * FROM read_parquet('/mnt/c/users/neils/projects/python-projects/f1-finishing-positions/.claude/worktrees/pipeline-foundation/data/features/*.parquet', filename = true);

-- Stage 2 sessionised driver-laps
CREATE OR REPLACE VIEW sessions AS
  SELECT * FROM read_parquet('/mnt/c/users/neils/projects/python-projects/f1-finishing-positions/.claude/worktrees/pipeline-foundation/data/sessions/*.parquet', filename = true);

-- Stage 1 raw endpoints (globbed across all races)
CREATE OR REPLACE VIEW raw_laps AS
  SELECT * FROM read_parquet('/mnt/c/users/neils/projects/python-projects/f1-finishing-positions/.claude/worktrees/pipeline-foundation/data/raw/*/laps.parquet', filename = true);
CREATE OR REPLACE VIEW raw_results AS
  SELECT * FROM read_parquet('/mnt/c/users/neils/projects/python-projects/f1-finishing-positions/.claude/worktrees/pipeline-foundation/data/raw/*/session_result.parquet', filename = true);
CREATE OR REPLACE VIEW raw_stints AS
  SELECT * FROM read_parquet('/mnt/c/users/neils/projects/python-projects/f1-finishing-positions/.claude/worktrees/pipeline-foundation/data/raw/*/stints.parquet', filename = true);
-- car_data is huge and may be mid-backfill; query with filters, not SELECT *.
CREATE OR REPLACE VIEW raw_car_data AS
  SELECT * FROM read_parquet('/mnt/c/users/neils/projects/python-projects/f1-finishing-positions/.claude/worktrees/pipeline-foundation/data/raw/*/car_data.parquet', filename = true);
