# domain

Shared value objects (`citation.py`, `events.py`, `enums.py`) and the Postgres schema (`pg/`).

Independent of `core` — nothing here imports config or a client. Alembic lives alongside the
models in `migrations/`; `domain.pg.Base.metadata` is its autogenerate target.

```bash
cd packages/domain
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
```
