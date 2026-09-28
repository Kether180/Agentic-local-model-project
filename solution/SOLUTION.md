# Second order references

## The problem

The chat endpoint answers questions from the user's documents and cites what it used. The model
writes `[S1]`, and the app turns that into a link to the chunk it came from: *"this PDF, page 2"*.

That is not enough when the chunk is itself quoting other research. Take the page the README
shows (PDF 4, page 2):

```
Waist circumference and waist-to-height ratio are recommended as adjunct measures5.
...
References
5. Ndiaye AB, Kowalczyk PT. Waist-to-height ratio as an adjunct screening measure. 2022.
```

If the model repeats that claim, the honest citation is **Ndiaye 2022**, the study the document
itself cites with the `5`, not just "page 2 of this PDF". Users need citations that resolve to
the original source.

## What I built

When the model cites a chunk, the app now looks inside that chunk for citation marks (`5`,
`1-3`, `[2,4]`), finds the entries they point to in **the same document's reference list**, and
works out which of them support **the sentence the model actually wrote**. The annotation then
names the original source:

```
before:  [S1] → "Test Second Order References 4.pdf p.2"
after:   [S1] → "[5] Ndiaye AB, Kowalczyk PT. Waist-to-height ratio as an adjunct screening
                 measure. ... (cited in Test Second Order References 4.pdf p.2)"
```

If the chunk has no citation marks, or the model's sentence doesn't match any cited sentence in
it, the annotation stays exactly as before.

- Parser: `apps/api/src/api/services/agent/references.py`
- Attribution: `apps/api/src/api/services/agent/citations.py`
- Tests: `tests/test_references.py` (17 unit tests) and `tests/test_samples.py` (the 5 sample
  PDFs), 22 new, 84 in total

## How it works

The work happens at two moments: **when the search runs** (steps 1–3, before the model
answers) and **after the model answers** (step 4). The search now runs as a fixed step on every
question instead of being left to the model (see Issues). The pipeline and the database are
unchanged.

```
  User's question
    │
    ▼
  Search the documents          ◄──  Postgres: chunks + vectors             existing,
    │                                                                        now always runs
    │
    ▼
  1. Rebuild the pages of the documents that were found                     NEW
  2. Read each document's reference list                                    NEW
  3. Find the citation marks in each chunk, with their sentence             NEW
    │
    ▼
  Model writes the answer, citing chunks as [S1], [S2] ...                  existing
    │
    ▼
  4. For each [Sn]: find the chunk sentence the model restated              NEW
     and cite that sentence's references
    │
    ▼
  Annotation: "[5] Ndiaye AB ... (cited in ... p.2)"
```

| Step | What it does | On the README page (PDF 4, page 2) |
|---|---|---|
| **1. Rebuild the pages** | A reference list is usually split across chunks, so the found chunk alone is not enough. All chunks of the documents found are loaded in one query and joined back into pages. | The page's reference list is split over two chunks: entries 1–2 in one, 2–5 in the next. Rebuilt, the page has all five. |
| **2. Read the reference list** | Find the `References` heading and read the numbered entries after it (`[12] ...` in journals, `12. ...` in slides). A journal list that continues onto the next page without a heading is followed. | `1. Halvorsen RT ...` to `5. Ndiaye AB ... Waist-to-height ratio as an adjunct screening measure. 2022` |
| **3. Find the citation marks** | Find the numbers that are citation marks and expand them. Keep each one with the sentence it belongs to. Skip numbers that only look like marks. | `health1-3` → 1, 2, 3 · `stature2,4` → 2, 4 · `adjunct measures5` → 5 · `29.9*`, `kg/m²` and the footnote's `populations5` are skipped |
| **4. Cite the right sources** | Compare the model's sentence before `[Sn]` with each cited sentence of the chunk. The one it clearly restates (at least two shared key words) gives the sources. No clear match: keep the old page citation. | *"Waist-to-height ratio is recommended as an adjunct measure [S1]"* shares *waist, height, ratio, recommended, adjunct, measure* with the sentence of mark 5 → **[5] Ndiaye** only |

## The most difficult part

**Telling real citation marks apart from ordinary numbers.** In the PDF a citation mark is a
small raised digit, but in the extracted text that formatting is gone: `health1-3` is just a word
followed by digits. The same page also has numbers that look exactly like citation marks and are
not, and the README names three of them:

| Looks like a citation mark | What it really is |
|---|---|
| `29.9*` | a decimal, with a footnote marker |
| `kg/m²` (extracted as `kg/m2` in two PDFs) | a unit |
| `populations5` inside a footnote | a real mark, but a third order reference (out of scope) |

Writing a special rule for every case would never end. The solution is one general rule: **a
number only counts as a citation mark if every number in it exists in that document's reference
list.** That removes decimals, page numbers and most stray digits at once. A few targeted rules
cover the rest: a digit right after a decimal point, a digit on a unit after a slash, and text
inside footnotes.

**Making sure the source supports the sentence.** A first version attached every source in the
chunk to `[S1]`, so a sentence about waist-to-height ratio also cited four other papers on the
same page that do not support it. Worse, when the model repeated the author's own finding, it pointed at other people's
work. Matching the model's sentence to the specific cited sentence fixed this. When no sentence
matches clearly, the app keeps the plain page citation, because a wrong source is worse than none.

## Requirements checklist

| README requirement | How it is met | Test |
|---|---|---|
| `1-3` → 1, 2, 3 and `2,4` → 2, 4 | Ranges (`-` and `–`) and lists expand | `test_ranges_and_lists_expand` |
| Several marks per chunk | Every mark keeps the sentence it is attached to | `test_each_mark_keeps_the_sentence_it_is_attached_to` |
| `29.9*` is not a mark | A digit after a decimal point is not a mark; `*` is never part of one | `test_superscript_marks_skip_the_readme_false_positives` |
| `kg/m²` is not a mark | `²` is not a digit; when extracted as `kg/m2`, digits on a unit after a slash are skipped | same, plus `test_unit_exponent_extracted_as_a_plain_digit_is_not_a_mark` |
| `populations5` in a footnote is third order | Footnotes are skipped, also when wrapped onto a second line | same, plus two footnote tests |
| Second order only, same document, no lookups | Marks resolve to entry text from the same PDF; no network calls | by design |
| Resolve to the original source | Only the sources of the sentence the model restated | `test_annotation_cites_only_the_sources_of_the_restated_sentence` |
| No marks → behaves as today | The original `url_citation`, unchanged | `test_chunk_without_marks_keeps_first_order_citation` + the 4 existing tests |

## Design decisions

| Decision | Why | Trade-off |
|---|---|---|
| Deterministic parsing, no LLM | Free, instant, testable, and it cannot invent a reference | Relies on text patterns; reading the PDF layout (font size, raised baseline) would find superscripts directly, but means changing the parse step |
| "Must exist in the list" validation | One rule instead of many special cases | A glued number that matches a real entry slips through (`CO2`, `phase3`) |
| Match the model's sentence, fall back to the page | A source must support the sentence | A heavy paraphrase falls back to the page citation: safe, less specific |
| Same-page list before the whole document's | Decks that restart numbering on each slide would otherwise cite the wrong paper | About 8 lines; no effect on the 5 samples |
| Resolve at question time, in the API | No pipeline change, migration or re-seed; works on existing data | Re-parses the retrieved documents on each question (about 4 ms measured). At scale this moves to ingest |
| Keep `url_citation`, the spec's only annotation type | No schema change for clients | Sources on the same page share a URL and differ by `title` |
| Search as a fixed graph step (`detect_filters → search → agent`), not the model's choice | Every answer is grounded in real results, whatever the model size; the prompt asked for it, the graph now guarantees it | One search per question even when a follow-up would not need it; the model can still call the tool to search again |

**What is guaranteed:** an annotation can only point at an entry listed in the same document,
with its text copied from the PDF. An invented `[S9]` is dropped, and the model never writes
titles, authors or DOIs itself.
**What is not:** that the source supports the sentence. Matching sentences by shared words is a
heuristic. It avoids the worst case (pinning the author's own claim on someone else's paper), but
a paraphrase can still match the wrong cited sentence if they share key words. When several
cited sentences tie, all their sources are kept: in one live run, a model sentence about BMI
matched several cited sentences of the same PDF 5 chunk equally and received nine or more
sources, among them [29]–[37].

## How to run

```bash
docker compose up -d --build
uv run python scripts/seed_pipeline.py samples --timeout 900
uv run pytest tests/test_references.py tests/test_samples.py -v
```

On 24 Sep the MinIO images could not be pulled in my environment, so I ran
[Adobe S3Mock](https://github.com/adobe/S3Mock) through an override file and left `compose.yml`
unchanged. If MinIO fails for you too:
`docker compose -f compose.yml -f compose.s3mock.yml up -d --build`. The default seed timeout of
300 s is too short on CPU, hence `--timeout 900`.

Then, in Swagger (`localhost:8000/docs`), `POST /responses` with
`{"model": "small", "input": "Search the documents: is waist-to-height ratio a better screening
measure than BMI? Cite your sources using the [S1] labels."}`. When the model cites a chunk with
marks, the annotation title reads like `[2] Okonkwo BA ... (cited in ... p.2)`.

## How I tested it

Four levels, from fastest to most realistic. Each one catches problems the previous one cannot.

**1. Unit tests: 17 tests, about 1 second.** Short synthetic texts modelled on the extracted
text of the sample PDFs: one test per README case (see the checklist above), plus the edge cases
found along the way (wrapped footnotes, `kg/m2`, letter-spaced headings, lists split across
chunks). The 62 existing tests still pass.

**2. The real sample PDFs: `tests/test_samples.py`, about 2 seconds, offline.** The 5 PDFs from
`samples/` go through the production path (PyMuPDF extraction as in the parse step, the
pipeline's own chunker, pages rebuilt from the chunks), and the test checks that every reference
list is complete and every citation mark resolves. This is the test that catches what
hand-written cases cannot: reintroducing the letter-spaced heading bug makes it fail on PDF 3.
Whenever I changed a rule (footnotes, units), I also compared every mark found on all 5 PDFs
before and after the change, to prove it removed only what it should:

| PDF | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| References found | 50 | 24 | 24 | 34 | 135 |
| Unresolved marks | 0 | 0 | 0 | 0 | 0 |

**3. Live in the running API, without the model.** Checked with a script against the running
stack: real search, page loading, resolution and attribution inside the API container, fed with model-style sentences, so the result does not
depend on how the small model behaves. On the README page (PDF 4 p2):

| Model sentence | Annotations |
|---|---|
| "Waist-to-height ratio is recommended as an adjunct measure [Sn]" | [5] Ndiaye only |
| "BMI performs inconsistently across ancestry groups and at the extremes of stature [Sn]" | [2] Okonkwo, [4] Takahashi |
| "Excess adipose accumulation can impair health [Sn]" | [1], [2], [3] |
| "The authors present a table of BMI bands [Sn]" | the old page citation: `…4.pdf p.2` |

**4. Through the chat, with the real model.** `POST /responses` in Swagger, every annotation
checked by hand: `[S2]` after a sentence on ancestry and stature resolved to [2] and [4]; `[S5]` after PDF 5's sentence on
visceral adiposity resolved to exactly its sources [38]–[41]; a vague sentence kept the page
citation; malformed labels (`[S1-S4]`) were ignored rather than guessed.

Levels 3 and 4 found two problems the unit tests could not: the `kg/m2` unit (level 3, fixed)
and tables breaking sentences (level 4, documented below).

**Also checked:** 84 tests passing (62 existing + 22 new), ruff clean, basedpyright 0 errors (60
warnings: the original 59, plus one from LangGraph's own types on the new `add_node` line).
**Cost, measured:** about 4 ms added per question (1.4 ms existing search, 0.9 ms loading the
pages, 3.1 ms parsing) and zero model calls.

## Issues found along the way

| Issue | Status |
|---|---|
| Chat answers stopped after two tokens: the ten retrieved chunks filled Ollama's default context window (4,096 tokens here, per `ollama ps`), ending with `done_reason: length` | Fixed: `num_ctx: 8192` in `OllamaChatSettings`, passed in `core/chat.py`; 3 of 3 runs then ended normally |
| `kg/m²` extracts as `kg/m2` in PDFs 1 and 5, and reference [2] exists, so a table header cited an unrelated paper | Fixed, found in the live check |
| Attaching every source of a chunk to every `[Sn]` mis-cites | Fixed with sentence matching |
| Tables break sentences: PDF 5's Table 1 sits inside a cited sentence, so a sentence about the table got that sentence's sources | Known, not fixed; needs layout-aware parsing |
| The embed step runs 2 to 3 times per document: `core/rabbit.py` computes inside the callback of a `BlockingConnection`, so a long embed blocks heartbeats and the message is redelivered | Documented, data is correct |
| Re-running the seed duplicates every document (`ingest()` always inserts a row) | Documented |
| `qwen3.5:0.8b` sometimes skipped its search tool (searching was the model's choice, asked for only in the prompt), then answered from memory with invented labels (`[S2]`, `[S4]`, `[SM1]`) that the range check in `parse_markers` correctly dropped, leaving no citations | Fixed: the search is a fixed graph step before the model answers. After the change, every live run searched and cited real results |
| When several cited sentences tie on shared words, the sources of all of them are kept, so one sentence can get many sources | Known; see next steps (break ties, or embedding similarity) |
| `qwen3.5:0.8b` sometimes copies raw marks instead of `[Sn]`, or paraphrases too loosely to match a cited sentence (page fallback) | Known small-model limit; a larger model restates claims more faithfully |

## Changes

| File | Change |
|---|---|
| `apps/api/src/api/services/agent/references.py` | **New.** The parser (199 lines, pure: text in, cited sentences out) |
| `apps/api/src/api/services/agent/citations.py` | Sentence matching, fallback to the page citation (+48 / −4) |
| `apps/api/src/api/crud/chunk.py` | `ReferenceSource`, `CitedClaim`, `ChunkHit.claims`, `page_texts()` (+39) |
| `apps/api/src/api/services/agent/tools.py` | Attach cited sentences after the search; the search body moved into `run_search()`, shared by the tool and the graph |
| `apps/api/src/api/services/agent/graph.py` | New fixed `search` step between `detect_filters` and `agent`; prompt reworded to match |
| `apps/api/src/api/routers/responses.py` | Docstring: the new flow |
| `tests/test_references.py` | **New.** 17 unit tests |
| `tests/test_samples.py` | **New.** The parser on the 5 sample PDFs (5 test cases) |
| `packages/core/src/core/config.py`, `chat.py` | Ollama context window 4,096 → 8,192 (+4) |
| `compose.s3mock.yml` | **New.** Optional local override: S3Mock instead of MinIO |
| `scripts/seed_pipeline.py` | Prints a line whenever a document moves to a new step (+6) |

The layering is respected (SQL in `crud/`, logic in `services/`, and `crud` never imports from
`services`). No pipeline, schema or migration changes, and `compose.yml` and `domain` are
untouched.

## Limitations and next steps

1. **Move to ingest.** Resolve once in the pipeline and store the cited sentences with the chunk
   (a JSONB column, or `reference` plus link tables). This removes the per-question parse and the
   need to rebuild pages from overlapping chunks.
2. **Layout-aware parsing.** Reading the superscript flag, font size and block layout from the
   PDF would largely close both known gaps: glued false positives outside this corpus, and
   tables breaking sentences.
3. **Better sentence matching.** Embedding similarity instead of shared words would catch
   paraphrases and break ties between similar cited sentences; the embedder is already in the
   request.
4. **Show sources to the model**, so a capable model can name the original study itself.
5. **Down-rank reference-list chunks**, which currently rank first for some queries.
6. **Fix the embed redelivery**: set a heartbeat, or run the work outside the callback.
7. **A bigger chat model.** With the search now a fixed step, every answer is grounded whatever
   the model; a stronger model (Claude on Bedrock, already supported by `core/chat.py`) would
   restate claims more faithfully, so fewer citations fall back to the page.
8. **Scale.** At 1,000 PDFs the question path still works: the HNSW index keeps search fast, and
   the reference parsing only reads the documents found for a question. Ingest is what slows
   first (embeddings on CPU), so it needs a GPU or hosted embeddings and more embed workers, plus
   items 1, 5 and 6 above.
