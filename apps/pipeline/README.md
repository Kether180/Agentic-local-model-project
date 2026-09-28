# pipeline

Document-processing workers. One worker per queue, chained:

```
q.parse ─► q.chunk ─► q.embed ─► q.citation ─► q.index ─► postgres+pgvector
```

```bash
uv run pipeline embed          # run one step locally
docker compose up -d --scale pipeline-embed=4
```

Every step caches its output on `(content hash, step, step version, step config)`, so a re-import
of unchanged bytes replays from `step_cache` instead of re-running the model. Add a step by
subclassing `PipelineStep`, implementing `compute()`, and registering it in `steps/__init__.py`.
