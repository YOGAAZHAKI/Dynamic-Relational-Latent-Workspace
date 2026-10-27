# Dynamic Relational Latent Workspace (DRLW)

A multi-agent research prototype in which agent outputs become **embedding-backed
nodes** in a live graph of **typed relationships** (SUPPORTS, CONTRADICTS,
REFINES, DEPENDS_ON, REDUNDANT). Each agent is fed a *relevant subgraph*
instead of raw chat history.

> Status: **complete (chunks 1-10).**
> Full pipeline: data models, mock/live LLMs, embeddings, FAISS, relation
> detection, graph, subgraph retrieval, orchestrator, Streamlit UI and an
> offline evaluation harness with baselines and an ablation.

## Requirements

- Python **3.10+** (developed on 3.12)
- No API key needed while `DLRW_MOCK_MODE=true` (the default) - fully offline.

## Install

```powershell
cd E:\Latent
python -m pip install -e ".[dev]"

# optional: Streamlit frontend
python -m pip install -e ".[ui]"
```

## Configure (optional)

```powershell
Copy-Item .env.example .env
```

Every setting has a working default; an empty `.env` still runs.
`DLRW_MOCK_MODE=true` keeps everything offline and free.

## Run the tests

```powershell
python -m pytest          # full suite
python -m pytest -v       # verbose
python -m pytest tests/test_orchestrator.py -v
python -m pytest tests/test_evaluation.py -v
```

## Run the offline demos (no API key, no cost)

```powershell
python scripts/demo_mock.py         # chunk 2: four agents
python scripts/demo_vectors.py      # chunk 3: embeddings + FAISS (mock embedder)
python scripts/demo_vectors.py --real   # same demo with all-MiniLM-L6-v2
python scripts/demo_relations.py    # chunk 4: typed relation classification
python scripts/demo_relations.py --llm  # LLM-shaped path via offline mock LLM
python scripts/demo_graph.py        # chunk 5: incremental graph + queries + save/load
python scripts/demo_retrieval.py    # chunk 6: query -> scored, stance-labeled subgraph
python scripts/demo_orchestrator.py # chunk 7: plan -> workers -> cited final answer
python scripts/demo_eval.py         # chunks 9-10: A/B/C/D report + reproducible files
```

`demo_eval.py` writes `results/evaluation_report.json` (full audit trail)
and `results/evaluation_report.md` (summary). In mock mode its metrics are
stable run-to-run.

## Streamlit UI

```powershell
python -m pip install -e ".[ui]"
streamlit run app.py
```

The UI lets you ask a question, inspect subtasks, retrieved contexts,
stance labels, contradictions, the final cited answer, and the live
workspace graph. Snapshots can be saved and loaded from the sidebar.

## Evaluation

The harness compares five systems on four hand-built cases:

| system | behaviour |
|---|---|
| A: llm_only | direct answer, no retrieval or graph |
| B: flat_context | every workspace unit in one untyped context (full-context baseline) |
| C: vector_rag | semantic top-k seeds only, no graph expansion |
| D: drlw | plan -> typed subgraph retrieval -> workers -> grounded synthesis |
| ablation: no_expansion | same orchestration as D, but graph expansion is disabled |

Metrics are deterministic lexical/ID checks, not an LLM judge:

- retrieval recall/precision,
- conflict recall (counter-evidence must reach the context),
- citation precision/recall (citations must be real workspace units),
- answer and grounded key-point recall,
- a weighted composite.

Representative offline result:

| system | retrieval R | conflict R | grounded key points | composite |
|---|---:|---:|---:|---:|
| B: flat_context | 1.00 | 1.00 | 0.92 | **0.95** |
| D: drlw | 0.69 | 1.00 | 0.50 | **0.79** |
| C: vector_rag | 0.69 | 0.75 | 0.67 | **0.77** |
| ablation: no_expansion | 0.62 | 0.75 | 0.50 | **0.72** |
| A: llm_only | 0.00 | 0.00 | 0.33 | **0.07** |

Interpretation:

- B is a full-context upper bound. On these tiny cases it is strong and
  cheap; it does not scale to workspaces with hundreds of units, so it is
  a reference point rather than the product architecture.
- D beats vector RAG and the no-expansion ablation on conflict recall and
  composite, while exposing a typed, auditable reasoning trace.
- D uses more total tokens here because it runs multiple agent contexts.
  Token efficiency is reported per system, not hidden.
- These are mock-mode scores against simple lexical metrics. They measure
  retrieval/grounding behaviour, not factual correctness.

## Folder structure

```
src/latent_workspace/
├── config.py              # all settings, read from env / .env
├── models/
│   ├── enums.py           # RelationshipType, InformationType, AgentRole, statuses
│   ├── information_unit.py# InformationUnit — what agents produce
│   ├── edge.py            # InformationEdge — typed relationships
│   └── reason.py          # Subtask, AgentResult, ReasoningContext, run results
├── embeddings/
│   ├── base.py            # Embedder protocol
│   ├── mock_embedder.py   # deterministic bag-of-words hasher (offline)
│   ├── st_embedder.py     # sentence-transformers (all-MiniLM-L6-v2)
│   └── __init__.py        # build_embedder() factory
├── relations/
│   ├── base.py            # RelationPrediction, threshold + UNRELATED gate
│   ├── mock_detector.py   # deterministic rule-based classifier (offline)
│   ├── llm_detector.py    # LLM classifier with strict JSON + retries
│   ├── nli_detector.py    # optional NLI cross-encoder (experimental)
│   └── __init__.py        # build_relation_detector() factory
├── vector/
│   ├── base.py            # VectorStore protocol, VectorHit
│   ├── faiss_store.py     # IndexFlatIP + normalization + id sidecar
│   └── __init__.py        # build_vector_store() with dim validation
├── graph/
│   ├── workspace.py       # KnowledgeGraph: add/query/persist/provenance
│   └── __init__.py        # build_workspace() + public exports
├── retrieval/
│   ├── base.py            # token estimates, cosine, type-weight validation
│   ├── subgraph.py        # SubgraphRetriever: seeds -> expand -> score -> budget
│   └── __init__.py        # build_retriever() factory
├── llm/
│   ├── base.py            # LLMClient interface + error types
│   ├── openai_compat.py   # any OpenAI-compatible API (incl. Ollama)
│   ├── mock.py            # deterministic offline client
│   └── structured.py      # JSON extraction + retry/validation
├── agents/
│   ├── prompts.py         # prompt templates for all four agents
│   ├── base.py            # BaseAgent: timing + crash-proof error handling
│   ├── planner.py         # query -> subtasks + hypotheses
│   ├── researcher.py      # subtask + context -> evidence/facts
│   ├── critic.py          # context -> counter-evidence/refinements
│   └── synthesizer.py     # context -> grounded final answer
├── orchestrator/
│   ├── orchestrator.py    # plan -> workers -> graph -> synthesis (chunk 7)
│   └── __init__.py        # Orchestrator, build_orchestrator, helpers
├── ui/
│   ├── session.py         # Streamlit-free session state (testable, chunk 8)
│   ├── app.py             # Streamlit presentation layer
│   └── __init__.py
└── evaluation/
    ├── dataset.py         # EvalCase/EvalUnit + default cases
    ├── systems.py         # A/B/C/D systems + no-expansion ablation
    ├── metrics.py         # deterministic retrieval/grounding metrics
    ├── harness.py         # run matrix, summaries, JSON/Markdown reports
    └── __init__.py
app.py                     # Streamlit entry point (streamlit run app.py)
tests/
├── conftest.py            # shared fixtures
├── test_models.py         # data-layer tests (chunk 1)
├── test_llm.py            # mock + OpenAI-compatible clients
├── test_structured_json.py# JSON parsing, retry, schema failures
├── test_agents.py         # all four agents, success + failure paths
├── test_embeddings.py     # mock + real embedder tests
├── test_vector_store.py   # FAISS add/search/remove/persist/validation
├── test_vector_integration.py  # embedder + store end-to-end
├── test_relations.py      # all 6 relation types, thresholds, LLM/NLI paths
├── test_graph.py          # KnowledgeGraph: add/query/provenance (chunk 5)
├── test_graph_persistence.py  # snapshot round-trip + corruption cases
├── test_retrieval.py      # SubgraphRetriever: ranking/expansion/budget (chunk 6)
├── test_retrieval_integration.py  # graph -> retriever -> agent pipeline
├── test_orchestrator.py   # full pipeline + failure handling (chunk 7)
├── test_ui_session.py     # UI state without Streamlit (chunk 8)
└── test_evaluation.py     # metrics, baselines, reports (chunks 9-10)
scripts/
├── demo_mock.py           # offline agent demo (chunk 2)
├── demo_vectors.py        # embeddings + FAISS (chunk 3)
├── demo_relations.py      # relation detection (chunk 4)
├── demo_graph.py          # knowledge-graph demo (chunk 5)
├── demo_retrieval.py      # subgraph retrieval demo (chunk 6)
├── demo_orchestrator.py   # end-to-end orchestration (chunk 7)
└── demo_eval.py           # evaluation + saved reports (chunks 9-10)
results/                   # generated reports (git-ignored)
```

## Design guarantees

- **Crash-proof orchestration:** a failed planner, worker, synthesizer or
  retrieval call is recorded in `errors`; the run continues where possible.
- **Grounded answers:** the synthesizer only sees retrieved contexts. If it
  fails, a deterministic extractive fallback cites real unit ids.
- **Conflicts stay visible:** `CONTRADICTS` has the same retrieval weight as
  `SUPPORTS`; budget cuts mark items as excluded and set `truncated`.
- **Auditable runs:** every context, subtask, unit, edge and metric is kept
  in `OrchestratorResult` / `EvalRun`.
- **Offline first:** mock mode needs no API key, network or spend.
- **Reproducible evaluation:** the harness substitutes deterministic ids for
  the duration of a run, then averages transparent per-case metrics.

## Key conventions (project-wide)

- **Edge direction:** `source --RELATIONSHIP--> target` reads
  *"source RELATIONSHIP target"* (e.g. `supply shock --SUPPORTS--> prices rose`).
  A newer unit (critic's evidence) points **at** the older unit it acts on.
- **`UNRELATED` is never stored** - it is a classifier outcome meaning
  "create no edge".
- **Embeddings are optional on the unit** (`embedding: list[float] | None`);
  units exist before they are vectorized (Chunk 3).
- **Timestamps are timezone-aware UTC.**
- **Ids:** `iu_xxxxxxxx` (units), `edge_xxxxxxxx`, `st_xxxxxxxx` (subtasks).
- **Predicted, not proven:** edges are confidence-scored predictions.
  Retrieval and answers must surface disagreement, not treat edges as truth.

## Roadmap

| Chunk | Content | Status |
|---|---|---|
| 1 | Scaffold + data models | done |
| 2 | LLM client (openai-compat + mock) and agent framework | done |
| 3 | Embeddings + FAISS vector store | done |
| 4 | Relationship detection (LLM/NLI/mock classifiers) | done |
| 5 | KnowledgeGraph: incremental node/edge updates, provenance | done |
| 6 | Reasoning-path (subgraph) retrieval | done |
| 7 | Orchestrator + final answer generation | done |
| 8 | Streamlit frontend | done |
| 9 | Evaluation harness (baselines A-D + ablation) | done |
| 10 | Documentation, reproducible experiments | done |

## Final audit (summary)
- Tests: **276 passed, 1 warning** (`python -m pytest`).
- Compile: `python -m compileall -q src scripts app.py` OK.
- Demos: all offline demos OK; orchestrator and evaluation run successfully.
- Workflow verified end-to-end (planner→researcher→critic→embeddings→FAISS→relations→graph→retrieval→synthesizer) with contradictions preserved and grounded citations.
- Streamlit: `streamlit run app.py` (entry `app.py`), UI shows agents/graph/paths/answers/metrics; imports OK.
- Persistence/config: save/load consistent; mock works offline; env vars (`DLRW_*`) work; `.env` gitignored.
- Evaluation: metrics match `results/evaluation_report.{json,md}`; formula/definitions documented; baselines comparable; ablation isolated to expansion; flat-context 0.95 reported honestly as upper bound. See `docs/AUDIT.md` for details.
