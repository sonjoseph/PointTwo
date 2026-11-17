BEGIN;
CREATE TABLE IF NOT EXISTS pt_workspaces (
 id text PRIMARY KEY, generation bigint NOT NULL DEFAULT 1,
 scorer jsonb NOT NULL, preparation_error text
);
CREATE TABLE IF NOT EXISTS pt_embedding_cache (
 model_id text NOT NULL, content_hash text NOT NULL, embedding vector(1024) NOT NULL,
 PRIMARY KEY(model_id, content_hash)
);
CREATE TABLE IF NOT EXISTS pt_entries (
 workspace_id text NOT NULL REFERENCES pt_workspaces(id), id uuid NOT NULL,
 external_id text NOT NULL, timekeeper text NOT NULL, input jsonb NOT NULL,
 revision bigint NOT NULL DEFAULT 1, source_revision bigint NOT NULL DEFAULT 1, state text NOT NULL DEFAULT 'waiting',
 accepted_id uuid, suggestions jsonb NOT NULL DEFAULT '[]',
 low_confidence boolean NOT NULL DEFAULT true, error text, run_id uuid,
 generation bigint, updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(workspace_id,id), UNIQUE(workspace_id,external_id),
 FOREIGN KEY(workspace_id,accepted_id) REFERENCES matters(workspace_id,id)
);
CREATE TABLE IF NOT EXISTS pt_entry_history (
 workspace_id text NOT NULL, entry_id uuid NOT NULL, revision bigint NOT NULL,
 input jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(workspace_id,entry_id,revision)
);
CREATE TABLE IF NOT EXISTS pt_runs (
 workspace_id text NOT NULL, id uuid NOT NULL, entry_id uuid, snapshot jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(workspace_id,id)
);
CREATE TABLE IF NOT EXISTS pt_decisions (
 id bigserial PRIMARY KEY, workspace_id text NOT NULL, entry_id uuid NOT NULL,
 entry_revision bigint NOT NULL, timekeeper text NOT NULL, normalized text NOT NULL,
 matter_id uuid, matter_revision bigint, action text NOT NULL CHECK(action IN ('accept','reject','undo')),
 run_id uuid, created_at timestamptz NOT NULL DEFAULT now(),
 FOREIGN KEY(workspace_id,entry_id) REFERENCES pt_entries(workspace_id,id)
);
CREATE INDEX IF NOT EXISTS pt_decisions_lookup ON pt_decisions(workspace_id,timekeeper,normalized,id DESC);
CREATE TABLE IF NOT EXISTS pt_vectors (
 workspace_id text NOT NULL, source_id uuid NOT NULL, kind text NOT NULL CHECK(kind IN ('catalog','example')),
 model_id text NOT NULL, source_revision bigint NOT NULL, matter_id uuid NOT NULL,
 matter_revision bigint NOT NULL, timekeeper text NOT NULL DEFAULT '', normalized text NOT NULL DEFAULT '',
 body text NOT NULL, embedding vector(1024) NOT NULL CHECK(vector_norm(embedding)>0),
 PRIMARY KEY(workspace_id,source_id,kind,model_id)
) PARTITION BY LIST(workspace_id);
COMMIT;
