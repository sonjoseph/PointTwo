BEGIN;
ALTER TABLE pt_workspaces ADD COLUMN IF NOT EXISTS decision_config jsonb;
COMMIT;
