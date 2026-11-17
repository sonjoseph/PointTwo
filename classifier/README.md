# Python classifier

- `models.py`: prompt fields and validation.
- `embeddings.py`: local Qwen inference and durable model/input cache.
- `store.py`: imports, revisions, catalog preparation, decisions, and confirmed examples.
- `engine.py`: independent catalog/example retrieval, candidate scoring, and immutable traces.
- `scoring.py`: seeded weights, features, contributions, and uncalibrated review policy.
- `projection.py`: bounded PCA views with frozen comparison axes and exact cosine distances.
- `api.py`: authenticated local review/debug API and durable preparation worker.
- `seed.py`: idempotent bundled fixture initialization; skips existing IDs and never approves expected labels.
- `evaluate.py`: isolated synthetic quality and feedback evaluation.
- `jev.py`: fixed-shortlist text requests, strict typed-response validation, and experimental Jev review gates.
- `jev_engine.py`: application Jev decisions, immutable probability/evidence traces, and stale-result checks after hosted inference.
- `runtime.py`: explicit pipeline opt-in and private environment/key-file configuration.
- `compare_jev.py`: paired weighted/Jev synthetic evaluation; offline by default, explicit `--execute` for hosted inference.
- `benchmark.py`: catalog recall benchmark against exact search in a selected prepared workspace.
- `train.py`: offline candidate logistic-regression training from pre-feedback snapshots.

`python -m classifier` listens on loopback port 8794. Set `DATABASE_URL` privately; `POINTTWO_DATA` defaults to `.pointtwo/python`, `POINTTWO_WORKSPACE` to `local-review`, and `OLLAMA_URL` to `http://127.0.0.1:11434`. One token authorizes the configured workspace; imported UserId only scopes timekeeper evidence. This is a local service, not a production multi-tenant authentication system.

The worker discovers unprepared catalog revisions and waiting entries in Postgres, so it resumes after restart. Query-time validation rejects stale vectors. Catalog and example indexes are separate partial HNSW indexes on physical workspace partitions; the planner can select exact search for small scopes. The debugger reports the selected catalog query plan.

Source revisions change when narratives are edited. Review revisions also change on decisions, preventing stale submissions; decision events and run snapshots remain immutable. Accepting/correcting an entry updates the label and example transactionally after local inference. Rejection does not train semantic negative weights. Reclassification preserves reviewed assignments.

Training is an offline candidate generator, not an automatic deployment system. It requires 100 eligible reviewed entries, uses pre-feedback features and later splits, and writes coefficients/metrics under `.pointtwo/training`. It does not activate weights or schedule itself. Production needs calibration, subgroup/release gates, and an external weekly scheduler.

Set `POINTTWO_DECISION=jev` plus a private `TYPESAFE_API_KEY` or `TYPESAFE_API_KEY_FILE` to enable hosted decisions. The default without explicit configuration is the local weighted baseline. Ignored `POINTTWO_DATA/runtime.json` can persist `decision_pipeline` and `typesafe_api_key_file` for the desktop launcher. Model/prompt/policy changes queue unapproved entries while preserving accepted decisions.
