BEGIN;
ALTER TABLE pt_workspaces ADD COLUMN IF NOT EXISTS active_model text;
COMMIT;
