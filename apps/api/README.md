# api

Document ingest and semantic search.

Layered `routers/` (HTTP) → `services/` (business logic) → `crud/` (SQL), with `schemas/` as the
wire contract. ORM rows never leave `crud/` un-converted, so a column rename cannot change the
shape of a response by accident.

| Route | What |
|---|---|
| `POST /documents` | upload a PDF → blob store, row, `q.parse`; returns 202 |
| `GET /documents` | recent documents |
| `GET /documents/{id}` | one document plus its per-step pipeline status |
| `DELETE /documents/{id}` | row, chunks, and blob |
| `GET /search?q=` | embed the query, kNN over chunks via pgvector |

```bash
uv run fastapi dev apps/api/src/api/main.py
```
