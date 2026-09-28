# Agentic RAG over PDFs, with local models

A retrieval-augmented chat backend that answers questions from your PDFs and **cites the original
study**, not just the page it read. It runs fully locally: an agent built with LangGraph, small
open models served by Ollama, Postgres with pgvector, and a five-step ingestion pipeline on
RabbitMQ.

```
"Waist-to-height ratio is recommended as an adjunct measure [S1]."

before:  [S1] → "Test Second Order References 4.pdf p.2"
after:   [S1] → "[5] Ndiaye AB, Kowalczyk PT. Waist-to-height ratio as an adjunct screening
                 measure. Clin Pract Advis. 2022 (cited in Test Second Order References 4.pdf p.2)"
```

## Highlights

- **Agent with a guaranteed search step.** A LangGraph graph (`detect_filters → search → agent`)
  always retrieves before the model answers, so every answer is grounded, even with a 0.8B model
  that would sometimes skip its search tool.
- **Second order references.** When the model cites a chunk that itself cites other research
  (`health1-3`, `stature2,4`, `[31]`), the citation resolves to the entry in that PDF's own
  reference list. Deterministic parsing, no LLM: it cannot invent a reference.
- **Local models by default.** Chat `qwen3.5:0.8b` and embeddings `qwen3-embedding:0.6b` on Ollama.
  Claude on AWS Bedrock is a configuration switch away.
- **Cached, scalable pipeline.** `parse → chunk → embed → citation → index`, one queue and one
  worker service per step, cached on content hash so re-ingesting a file is free.
- **OpenResponses-style API.** `POST /responses` returns the answer with `url_citation`
  annotations, as JSON or a server-sent event stream.

## Architecture

```
                         ┌──────────────── API (FastAPI) ────────────────┐
  Client ── HTTP ──►     │  POST /responses                               │
                         │   detect_filters → search → agent → citations  │ ──► Ollama (chat + embeddings)
                         │  GET /search · /documents                      │ ──► Postgres + pgvector
                         └──────────────┬─────────────────────────────────┘
                                        │ upload
                                        ▼
                    RabbitMQ ──► parse → chunk → embed → citation → index ──► Postgres
                                  (5 workers)                  ▲
                    Blob store (MinIO / S3Mock) ───────────────┘ the PDFs
```

### How a question is answered

| Step | Where | What happens |
|---|---|---|
| Detect filters | `services/agent/graph.py`, `filters.py` | One chat call: does the question name an author, year or journal? |
| **Search** (always) | `graph.py` `_search` → `tools.py` `run_search()` | Embed the question, vector search over chunks in Postgres |
| Resolve references | `crud/chunk.py` `page_texts()`, `services/agent/references.py` | Rebuild the pages, read each reference list, find citation marks and their sentences |
| Answer | `graph.py` (`create_agent`) | The model answers from results labelled `[S1]…[S10]`; it may search again |
| Cite | `services/agent/citations.py` | Each `[Sn]` becomes an annotation: the original study, or the page if no cited sentence matches |

### Data model

Four tables, defined once in `packages/domain/src/domain/pg/models.py` and created by Alembic:
`document` (one per PDF), `chunk` (text + 1,024-dimension vector, HNSW index), `step_cache`
(each pipeline step's result) and `pipeline_status` (per-step progress log).

## Quickstart

Requirements: Docker with Compose, and optionally [uv](https://docs.astral.sh/uv/) for local
tests.

```bash
docker compose up -d --build                                   # pulls ~3 GB of models on first run
uv sync                                                        # optional: local venv for tests and LSPs
uv run python scripts/seed_pipeline.py samples --timeout 900   # ingest the 5 sample PDFs
```

If the MinIO images cannot be pulled, use the S3Mock override instead:
`docker compose -f compose.yml -f compose.s3mock.yml up -d --build`.

| | |
|---|---|
| API docs (Swagger) | http://localhost:8000/docs |
| RabbitMQ | http://localhost:15672 (guest/guest) |

Ask a question:

```bash
curl -s localhost:8000/responses -H 'content-type: application/json' -d '{
  "model": "small",
  "input": "Is waist-to-height ratio a better screening measure than BMI? Cite your sources using the [S1] labels."
}'
```

A second order annotation looks like:

```json
{"type": "url_citation", "url": "/documents/…#page=2",
 "title": "[2] Okonkwo BA, Lindqvist H. Limitations of body mass index as a compositional proxy. … (cited in Test Second Order References 4.pdf p.2)",
 "start_index": 184, "end_index": 188}
```

## API

| | |
|---|---|
| `POST /documents` | Upload a PDF: blob store, database row, first queue. Returns 202 |
| `GET /documents` | Recent documents |
| `GET /documents/{id}` | One document with its per-step pipeline status |
| `DELETE /documents/{id}` | Row, chunks and blob |
| `GET /search?q=` | Embed the query, nearest chunks via pgvector |
| `POST /responses` | Answer with citations; `"stream": true` for server-sent events |

Layered `routers/` (HTTP) → `services/` (logic) → `crud/` (SQL), with `schemas/` as the wire
contract.

## Second order references

The sample PDFs (`samples/`, five generated documents: journal articles and slide decks) use every
common citation style: superscripts glued to a word (`health1-3`), after a full stop
(`per cent.1–3`), and bracketed (`[1,2]`), with ranges and lists. They also contain look-alikes that
must not count:

![A sample page annotated with citation marks, false positives and a third order reference](figs/second_order_take_home.jpg)

| Looks like a citation | What it is |
|---|---|
| `29.9*` | a decimal with a footnote marker |
| `kg/m²` (extracted as `kg/m2`) | a unit |
| `populations5` inside a footnote | a third order reference, out of scope |

The core rule: **a number is only a citation mark if every number in it exists in that document's
reference list**, plus a few targeted rules for decimals, units and footnotes. The model's sentence
is then matched to the cited sentence it restates (shared key words), so a citation only carries
the sources of the claim it supports. With no clear match it keeps the plain page citation: a
wrong source is worse than none.

Full write-up, design decisions, testing and limitations: [`solution/SOLUTION.md`](solution/SOLUTION.md).

## Configuration

Settings are `APP_`-prefixed environment variables, nested with `__` (see `core.config.Settings`
and `compose.yml`). `APP_CHAT__PROVIDER` and `APP_EMBEDDING__PROVIDER` each take `ollama`
(default) or `bedrock`, independently.

To use Claude on Bedrock, add a compose override that mounts your AWS credentials:

```yaml
# bedrock.yml
x-aws: &aws
  volumes:
    - ~/.aws:/home/nonroot/.aws:ro
  environment:
    APP_CHAT__PROVIDER: bedrock
    APP_CHAT__REGION: eu-central-1
    APP_EMBEDDING__PROVIDER: bedrock
    APP_EMBEDDING__REGION: eu-central-1
    AWS_PROFILE: <your profile>
    AWS_REGION: eu-central-1

services:
  api:               {<<: *aws}
  pipeline-citation: {<<: *aws}
  pipeline-embed:    {<<: *aws}
```

Then run `docker compose -f compose.yml -f bedrock.yml up -d`. The embedding step, the slowest,
scales on its own: `docker compose up -d --scale pipeline-embed=4`.

## Tests

```bash
uv run pytest             # 84 tests, offline: no Docker, broker or model needed
uv run ruff check .
uv run basedpyright
```

`tests/test_samples.py` runs the 5 sample PDFs through the production path (PyMuPDF extraction,
the pipeline's chunker, pages rebuilt from chunks) and checks that every reference list is complete
and every citation mark resolves.

## Project structure

```
.
├── apps
│   ├── api          FastAPI: ingest, semantic search, the agent and chat
│   └── pipeline     five step workers, one per queue
├── packages
│   ├── core         config, chat/embedding/blob/queue clients, hashing, sessions
│   └── domain       value objects, the Postgres schema (SQLAlchemy), Alembic migrations
├── samples          five sample PDFs
├── scripts          seed_pipeline.py: push a folder of PDFs through, print a report
├── solution         design write-up
└── tests            offline tests
```

## Credits

Built on a starter codebase provided for an engineering exercise (the ingestion pipeline, API and
agent scaffold). My work: second order reference resolution (`references.py`, `citations.py`,
`crud/chunk.py`), the fixed search step in the agent graph, the chat model context-window fix, the
S3Mock override, and the tests in `tests/test_references.py` and `tests/test_samples.py`.
