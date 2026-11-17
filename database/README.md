# Postgres and pgvector

PointTwo uses Postgres with the `vector` extension. Set `DATABASE_URL` privately to a database provisioned for the application; infrastructure and credentials are not included in this repository. The database administrator must enable pgvector before startup. The application role needs permission to create tables, partitions, and indexes, but does not need to be a superuser.

The tested local configuration is Postgres 17 / pgvector 0.8.7 on a loopback connection. Application startup applies the numbered SQL schema files idempotently.

| Table | Content |
| --- | --- |
| `matters` | Catalog records, prompt metadata, revisions |
| `pt_entries` | Original inputs, suggestions, review state, accepted assignments |
| `pt_vectors` | Catalog and confirmed-example vectors, partitioned by workspace |
| `pt_embedding_cache` | Reusable model/input embeddings |
| `pt_decisions`, `pt_entry_history` | Append-only decisions and prior source inputs |
| `pt_runs` | Immutable scoring/evidence snapshots |
| `pt_workspaces` | Persisted scorer and evidence generation |

Useful read-only queries:

```sql
SELECT workspace_id, count(*) FROM pt_entries GROUP BY workspace_id;
SELECT kind, count(*) FROM pt_vectors WHERE workspace_id='local-review' GROUP BY kind;
SELECT scorer FROM pt_workspaces WHERE id='local-review';
SELECT id, snapshot->'timing_ms' FROM pt_runs
WHERE workspace_id='local-review' ORDER BY created_at DESC LIMIT 5;
SELECT indexname, indexdef FROM pg_indexes WHERE tablename LIKE 'pt_v_%';
```

Each workspace partition has separate cosine HNSW indexes for catalog and examples. Search uses `ef_search=400`, iterative scans, and revision/scope checks. Small scopes may use exact search; the debugger records the catalog query plan. Inspect `EXPLAIN` and measure recall instead of assuming index use or quality.

The first schema includes `embedding_models` and `matter_embeddings` for compatibility with previously initialized databases. The application and benchmark use `pt_vectors`; applied schema history is retained, and startup does not remove existing data. Schema/index changes need deliberate migrations: `CREATE INDEX IF NOT EXISTS` does not retune an existing index.

The local token authorizes one configured workspace. Database partitioning alone is not production multi-tenant authorization.
