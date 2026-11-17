-- Application schema. Enable the vector extension as an administrator first.
BEGIN;
CREATE TABLE IF NOT EXISTS embedding_models (
    id text PRIMARY KEY,
    dimensions integer NOT NULL CHECK (dimensions = 1024)
);
CREATE TABLE IF NOT EXISTS matters (
    workspace_id text NOT NULL,
    id uuid NOT NULL,
    client_id text NOT NULL,
    matter_id text NOT NULL,
    client_name text NOT NULL,
    matter_name text NOT NULL,
    description text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}',
    status text NOT NULL CHECK (status IN ('Open', 'Closed')),
    revision bigint NOT NULL CHECK (revision > 0),
    PRIMARY KEY (workspace_id, id),
    UNIQUE (workspace_id, client_id, matter_id)
);
CREATE TABLE IF NOT EXISTS matter_embeddings (
    workspace_id text NOT NULL,
    matter_id uuid NOT NULL,
    model_id text NOT NULL REFERENCES embedding_models(id),
    source_revision bigint NOT NULL,
    body text NOT NULL,
    embedding vector(1024) NOT NULL CHECK (vector_norm(embedding) > 0),
    PRIMARY KEY (workspace_id, matter_id, model_id),
    FOREIGN KEY (workspace_id, matter_id) REFERENCES matters(workspace_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS matter_embeddings_hnsw
    ON matter_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 32, ef_construction = 256);
CREATE INDEX IF NOT EXISTS matter_embeddings_scope
    ON matter_embeddings (workspace_id, model_id);
COMMIT;
