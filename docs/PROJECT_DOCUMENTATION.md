# Dynamic Relational Latent Workspace for Structured Multi-Agent Reasoning - Complete Technical Documentation

**Version:** 0.1.0  
**Status:** Complete (Chunks 1-10)  
**Last Verified:** 276 tests passed, 1 warning (see [docs/AUDIT.md](AUDIT.md))  

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Project Folder Structure](#4-project-folder-structure)
5. [Core Components Explained](#5-core-components-explained)
6. [End-to-End Example](#6-end-to-end-example)
7. [Setup and Installation](#7-setup-and-installation)
8. [How to Run the Project](#8-how-to-run-the-project)
9. [Testing and Verification](#9-testing-and-verification)
10. [Evaluation and Research](#10-evaluation-and-research)
11. [Configuration and Persistence](#11-configuration-and-persistence)
12. [Troubleshooting](#12-troubleshooting)
13. [Limitations and Future Work](#13-limitations-and-future-work)
14. [New Developer Onboarding](#14-new-developer-onboarding)

## 1. Project Overview

### What the project does
The Dynamic Relational Latent Workspace (DRLW) is a multi-agent research prototype where agent outputs become embedding-backed nodes in a live graph of typed relationships (SUPPORTS, CONTRADICTS, REFINES, DEPENDS_ON, REDUNDANT). Each agent is fed a *relevant subgraph* instead of raw chat history.

### Why we built it
Traditional multi-agent systems often pass large chat histories between agents, leading to context bloat and lost provenance. DRLW addresses this by building a structured, queryable knowledge graph that accumulates evidence with explicit relationships.

### What problem it solves
- **Context management**: Instead of dumping history, agents receive targeted subgraphs
- **Provenance**: Every claim can be traced through typed relationships
- **Contradiction handling**: Counter-evidence is explicitly surfaced, not hidden
- **Grounding**: Final answers cite real workspace units

### Main idea and expected benefits
The core idea is that reasoning should happen over a **relational workspace** where information units are connected by typed, confidence-scored edges. Benefits include better traceability, explicit handling of conflicting evidence, and more grounded, auditable answers.

## 2. System Architecture

### Complete workflow from user query to final answer

`mermaid
flowchart TD
    A[User Query] --> B[Orchestrator]
    B --> C[Planner Agent]
    C --> D[Subtasks Created]
    D --> E[For each subtask]
    E --> F[Reasoning-Path Retrieval]
    F --> G[Researcher or Critic Agent]
    G --> H[Agent produces InformationUnits]
    H --> I[Embed units]
    I --> J[FAISS search for candidates]
    J --> K[Relationship Detection]
    K --> L[Update Dynamic Graph]
    L --> M{More rounds/subtasks?}
    M -->|Yes| E
    M -->|No| N[Synthesizer Agent]
    N --> O[Final Answer with Citations]
`

### Component connections
- **Agents** ([src/latent_workspace/agents/](../src/latent_workspace/agents/)): Planner, Researcher, Critic, Synthesizer - all extend BaseAgent
- **LLM** ([src/latent_workspace/llm/](../src/latent_workspace/llm/)): OpenAI-compatible or Mock client
- **Embeddings** ([src/latent_workspace/embeddings/](../src/latent_workspace/embeddings/)): Generate vectors for units/queries
- **Vector Search** ([src/latent_workspace/vector/](../src/latent_workspace/vector/)): FAISS store for similarity search
- **Relationship Detector** ([src/latent_workspace/relations/](../src/latent_workspace/relations/)): Classify relationships between units
- **Graph** ([src/latent_workspace/graph/](../src/latent_workspace/graph/)): KnowledgeGraph stores nodes/edges with provenance
- **Retrieval** ([src/latent_workspace/retrieval/](../src/latent_workspace/retrieval/)): SubgraphRetriever builds reasoning paths
- **Orchestrator** ([src/latent_workspace/orchestrator/](../src/latent_workspace/orchestrator/)): Coordinates the full workflow

## 3. Technology Stack

### Core dependencies (from pyproject.toml)
| Technology | Purpose | Why chosen |
|---|---|---|
| pydantic>=2.6 | Data validation/models | Type-safe models with v2 features |
| pydantic-settings>=2.2 | Config from env/.env | Clean settings management |
| python-dotenv>=1.0 | .env loading | Standard env file support |
| openai>=1.30 | OpenAI-compatible API client | Works with OpenAI, Ollama, OpenRouter, etc. |
| numpy>=1.26 | Vector operations | Fast numerical computing |
| faiss-cpu>=1.8 | Vector similarity search | Efficient nearest neighbor search |
| sentence-transformers>=3.0 | Local embeddings | Optional real embeddings (all-MiniLM-L6-v2) |

### Optional components
| Component | Install | Purpose |
|---|---|---|
| Dev tools | pip install -e ".[dev]" | pytest for testing |
| UI | pip install -e ".[ui]" | Streamlit frontend |

## 4. Project Folder Structure

Actual repository structure (from inspection):

```
.env.example
.gitignore
app.py
pyproject.toml
README.md
docs/
  AUDIT.md
scripts/
  demo_eval.py, demo_graph.py, demo_mock.py, demo_orchestrator.py,
  demo_relations.py, demo_retrieval.py, demo_vectors.py
src/latent_workspace/
  __init__.py, config.py
  agents/ (base.py, planner.py, researcher.py, critic.py, synthesizer.py, prompts.py, __init__.py)
  embeddings/ (base.py, mock_embedder.py, st_embedder.py, __init__.py)
  evaluation/ (dataset.py, systems.py, metrics.py, harness.py, __init__.py)
  graph/ (workspace.py, __init__.py)
  llm/ (base.py, mock.py, openai_compat.py, structured.py, __init__.py)
  models/ (enums.py, information_unit.py, edge.py, reason.py, __init__.py)
  orchestrator/ (orchestrator.py, __init__.py)
  relations/ (base.py, mock_detector.py, llm_detector.py, nli_detector.py, __init__.py)
  retrieval/ (base.py, subgraph.py, __init__.py)
  ui/ (session.py, app.py, __init__.py)
  vector/ (base.py, faiss_store.py, __init__.py)
tests/ (20 test files including test_orchestrator.py, test_evaluation.py, test_ui_session.py)
results/ (generated reports)
```

Key files: `app.py` (Streamlit entry), `src/latent_workspace/config.py` (settings), `src/latent_workspace/orchestrator/orchestrator.py` (workflow), `src/latent_workspace/graph/workspace.py` (graph), `src/latent_workspace/retrieval/subgraph.py` (reasoning paths).

## 5. Core Components Explained

### LLM client and mock client
- Base: [src/latent_workspace/llm/base.py](../src/latent_workspace/llm/base.py) defines LLMClient interface and errors.
- Mock: [src/latent_workspace/llm/mock.py](../src/latent_workspace/llm/mock.py) provides deterministic offline responses.
- OpenAI-compatible: [src/latent_workspace/llm/openai_compat.py](../src/latent_workspace/llm/openai_compat.py) for real APIs.
- Structured: [src/latent_workspace/llm/structured.py](../src/latent_workspace/llm/structured.py) for JSON extraction/retry.

### Planner, Researcher, Critic, Synthesizer
- Base: [src/latent_workspace/agents/base.py](../src/latent_workspace/agents/base.py) - crash-proof error handling, timing.
- Planner: creates subtasks/hypotheses ([src/latent_workspace/agents/planner.py](../src/latent_workspace/agents/planner.py))
- Researcher: gathers evidence from context ([src/latent_workspace/agents/researcher.py](../src/latent_workspace/agents/researcher.py))
- Critic: finds counter-evidence/refinements ([src/latent_workspace/agents/critic.py](../src/latent_workspace/agents/critic.py))
- Synthesizer: produces grounded final answer ([src/latent_workspace/agents/synthesizer.py](../src/latent_workspace/agents/synthesizer.py))
- Prompts: [src/latent_workspace/agents/prompts.py](../src/latent_workspace/agents/prompts.py)

### InformationUnit and Edge
- Unit: [src/latent_workspace/models/information_unit.py](../src/latent_workspace/models/information_unit.py) - content, type, agent, confidence, embeddings (optional)
- Edge: [src/latent_workspace/models/edge.py](../src/latent_workspace/models/edge.py) - typed relationship source->target, confidence, rationale
- Enums/Reason: [src/latent_workspace/models/enums.py](../src/latent_workspace/models/enums.py), [src/latent_workspace/models/reason.py](../src/latent_workspace/models/reason.py)

### Embeddings and FAISS
- Embedders: [src/latent_workspace/embeddings/](../src/latent_workspace/embeddings/) (mock deterministic, sentence-transformers)
- Vector store: [src/latent_workspace/vector/faiss_store.py](../src/latent_workspace/vector/faiss_store.py) - IndexFlatIP with normalization, id sidecar

### Relationship detection
- [src/latent_workspace/relations/](../src/latent_workspace/relations/): mock rules, LLM classifier, optional NLI. Threshold from settings (`relation_threshold`, default 0.65). UNRELATED not stored.

### Graph, persistence, provenance
- [src/latent_workspace/graph/workspace.py](../src/latent_workspace/graph/workspace.py): KnowledgeGraph with add_unit/add_edge, find_supporting/find_contradicting, provenance. Save/load snapshots (graph.json + graph.faiss).

### Retrieval and ReasoningContext
- [src/latent_workspace/retrieval/subgraph.py](../src/latent_workspace/retrieval/subgraph.py): seeds from query, BFS expansion, scoring, budget. Preserves contradictions. Returns ReasoningContext.

### Orchestrator, UI, SQLite
- Orchestrator: [src/latent_workspace/orchestrator/orchestrator.py](../src/latent_workspace/orchestrator/orchestrator.py)
- UI: [src/latent_workspace/ui/session.py](../src/latent_workspace/ui/session.py) (testable), [src/latent_workspace/ui/app.py](../src/latent_workspace/ui/app.py) (Streamlit)
- SQLite: settings define `db_path` (data/workspace.sqlite3) - present as storage path in config; implementation focus is graph/FAISS persistence.

## 6. End-to-End Example

**Sample query:** "Why did global chip prices rise in 2025?" (from demo/orchestrator)

**Workflow (illustrative with actual behavior):**
1. **Orchestrator** receives query, calls Planner
2. **Planner** creates subtasks (research + critique) and hypotheses ? added to graph as InformationUnits
3. **For each subtask:** Reasoning-Path Retrieval builds context; Researcher/Critic run and produce units
4. **Embeddings:** units embedded (mock or real); FAISS stores vectors
5. **Candidate retrieval:** graph finds similar units by vector search
6. **Relationship Detection:** pairs classified (SUPPORTS/CONTRADICTS/REFINES/DEPENDS_ON/REDUNDANT) with confidence
7. **Graph updates:** valid edges added (UNRELATED/low confidence skipped)
8. **Synthesizer:** receives final retrieved context, produces cited answer

**Note:** Specific unit IDs and exact counts vary by run (IDs are generated). The audit showed grounded citations and contradictions preserved. See [scripts/demo_orchestrator.py](../scripts/demo_orchestrator.py) for a runnable example.

## 7. Setup and Installation

### Prerequisites
- Python >= 3.10
- Windows PowerShell (commands below verified)

### Python environment setup
```powershell
cd E:\Latent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Dependency installation
```powershell
python -m pip install -e ".[dev]"
# Optional UI
python -m pip install -e ".[ui]"
```

### Environment configuration
```powershell
Copy-Item .env.example .env
```
Edit `.env` as needed. See [Configuration](#11-configuration-and-persistence).

### Mock mode and live LLM configuration
- **Mock mode (default):** `DLRW_MOCK_MODE=true` - no API keys, offline, deterministic.
- **Live mode:** Set `DLRW_MOCK_MODE=false`, configure `DLRW_LLM_BASE_URL`, `DLRW_LLM_API_KEY`, `DLRW_LLM_MODEL`. Works with OpenAI-compatible endpoints.

## 8. How to Run the Project

### Launch Streamlit
```powershell
streamlit run app.py
```

### Run offline demos
```powershell
python scripts/demo_mock.py
python scripts/demo_vectors.py
python scripts/demo_relations.py
python scripts/demo_graph.py
python scripts/demo_retrieval.py
python scripts/demo_orchestrator.py
```

### Run full test suite
```powershell
python -m pytest
python -m pytest -v
```

### Run evaluation scripts
```powershell
python scripts/demo_eval.py
python scripts/demo_eval.py --out-dir results
```

Expected outputs: demos print completion messages; tests show pass/fail; eval writes `results/evaluation_report.json` and `results/evaluation_report.md`.

## 9. Testing and Verification

### Test structure
Tests in [tests/](../tests/): models, llm, agents, embeddings, vector store, relations, graph, retrieval, orchestrator, UI session, evaluation (20 files). Run with pytest.

### Latest verified result
**276 passed, 1 warning** (from [docs/AUDIT.md](AUDIT.md), run via `python -m pytest`). Warning is from torch/jit in nli-related paths.

### What tests verify
- Data models, agents, LLM clients, embeddings, FAISS, relations, graph operations, retrieval, orchestration, UI state, evaluation metrics/harness.

### What they do not prove
- Live API behavior without keys/network, large-scale performance, factual correctness (metrics are lexical/ID-based), statistical significance.

References: [docs/AUDIT.md](AUDIT.md), [results/evaluation_report.json](../results/evaluation_report.json), [results/evaluation_report.md](../results/evaluation_report.md).

## 10. Evaluation and Research

### Baselines and ablation
Systems (from [src/latent_workspace/evaluation/systems.py](../src/latent_workspace/evaluation/systems.py)): A (llm_only), B (flat_context - all units untyped), C (vector_rag - semantic top-k only), D (drlw - full orchestration), ablation (no_expansion - expansion_depth=0). Same cases/settings.

### Metrics and composite formula (from code)
Metrics: retrieval_recall, retrieval_precision, conflict_recall, citation_precision, citation_recall, answer_key_point_recall, grounded_key_point_recall, composite.  
Formula: `composite = 0.35*retrieval_recall + 0.15*retrieval_precision + 0.20*conflict_component + 0.20*grounded_key_point_recall + 0.10*citation_precision` (rounded to 4 decimals).  
`conflict_component = 1.0` if no conflict units, else `conflict_recall`. Deterministic substring/ID matching, no judge model.

### Reported scores (from results/evaluation_report.json)
| system | composite | retrieval_recall | conflict_recall | citation_precision | grounded_key_point_recall |
|---|---|---|---|---|---|
| A: llm_only | 0.0670 | 0.0000 | 0.0000 | 0.0000 | 0.3333 |
| B: flat_context | 0.9533 | 1.0000 | 1.0000 | 1.0000 | 0.9167 |
| C: vector_rag | 0.7740 | 0.6875 | 0.7500 | 1.0000 | 0.6667 |
| D: drlw | 0.7906 | 0.6875 | 1.0000 | 1.0000 | 0.5000 |
| ablation: no_expansion | 0.7187 | 0.6250 | 0.7500 | 1.0000 | 0.5000 |

B is a full-context upper bound (not scalable) - reported honestly. D beats C and ablation on conflict recall and composite. No statistical significance claimed.

## 11. Configuration and Persistence

### Environment variables ([src/latent_workspace/config.py](../src/latent_workspace/config.py))
Prefix `DLRW_`. Key vars: `MOCK_MODE`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `TOP_K`, `COSINE_FLOOR`, `RELATION_THRESHOLD`, `MAX_ROUNDS`, `MAX_CONTEXT_TOKENS`, `DATA_DIR`. See [.env.example](../.env.example).

### API keys
Only needed when `DLRW_MOCK_MODE=false`. Stored in `.env` (gitignored). Never committed.

### Graph snapshots, FAISS index, metadata
- Snapshots: `graph.json` + `graph.faiss` via `KnowledgeGraph.save/load`. Atomic write for JSON.
- FAISS: `FaissVectorStore` with normalization, id sidecar.
- Metadata: units/edges have timestamps, types, agents, confidence, rationale.

### Experiment records
Evaluation results written to `results/` as JSON and Markdown. Harness uses deterministic IDs per run.

### Reset local data safely
```powershell
Remove-Item -Recurse -Force data, results -ErrorAction SilentlyContinue
```
Or use UI reset.

## 12. Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Import errors | Missing deps | `pip install -e ".[dev]"` |
| Streamlit not found | UI extra not installed | `pip install -e ".[ui]"` |
| FAISS issues | Platform/build | Use faiss-cpu (included). Ensure compatible Python.
| Model download slow | sentence-transformers first run | Use mock mode or pre-download; cached after first run.
| API errors in live mode | Bad key/url/model | Check `.env`, use OpenAI-compatible endpoint.
| Snapshot load fails | Version/dim mismatch | Use same embedder/dim as saved snapshot.
| Tests fail | Environment changes | Run from repo root, ensure deps installed.

## 13. Limitations and Future Work

### Implemented vs planned
Implemented: full pipeline through chunks 1-10 (models, agents, embeddings, FAISS, relations, graph, retrieval, orchestrator, UI, evaluation). Future improvements possible but not required.

### Important clarifications
- **Embedding similarity ? logical relationships.** Similarity is used for retrieval/candidates; relationship detection uses rules/LLM/NLI with thresholds.
- **Predicted edges ? factual truth.** Edges are confidence-scored predictions. System surfaces contradictions and provenance.
- **Sentence embeddings ? LLM hidden states.** Embeddings come from separate embedder models (mock or sentence-transformers), not the chat model's internal representations.

## 14. New Developer Onboarding

### First-day setup checklist
- [ ] Clone repo, create venv, activate
- [ ] Install `pip install -e ".[dev]"`
- [ ] Copy `.env.example` to `.env` (mock mode OK)
- [ ] Run `python -m pytest` - expect 276 passed
- [ ] Run a demo: `python scripts/demo_orchestrator.py`
- [ ] Try UI: `streamlit run app.py`

### Recommended reading order
1. [README.md](../README.md)
2. [docs/AUDIT.md](AUDIT.md)
3. [src/latent_workspace/models/](../src/latent_workspace/models/) (data models)
4. [src/latent_workspace/agents/](../src/latent_workspace/agents/) (agents)
5. [src/latent_workspace/graph/workspace.py](../src/latent_workspace/graph/workspace.py) (graph)
6. [src/latent_workspace/retrieval/subgraph.py](../src/latent_workspace/retrieval/subgraph.py) (retrieval)
7. [src/latent_workspace/orchestrator/orchestrator.py](../src/latent_workspace/orchestrator/orchestrator.py) (workflow)
8. [src/latent_workspace/evaluation/](../src/latent_workspace/evaluation/) (eval)

### How to run a small change and test it
1. Make change in src/
2. Run relevant tests: `python -m pytest tests/test_*.py -v`
3. Run full suite: `python -m pytest`
4. Check compile: `python -m compileall -q src`

### Rules for preserving architecture and tests
- Keep existing patterns (factories, protocols, crash-proof handling)
- Don't modify evaluation results or tests without justification
- Maintain offline-first behavior in mock mode
- Respect edge direction convention and UNRELATED handling
- All tests must still pass

