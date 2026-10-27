# Final End-to-End Audit Report (DRLW)

Date: 2026-10-27 (run in environment)

## 1) Verification commands (executed)
- `python -m pytest` → **276 passed, 1 warning** (47.81s). Warnings only from torch/nli-related paths.
- `python -m compileall -q src scripts app.py` → **OK**.
- Offline demos: `demo_mock.py`, `demo_vectors.py`, `demo_relations.py`, `demo_graph.py`, `demo_retrieval.py` all printed "OK … zero API calls" (or completed). `demo_orchestrator.py` completed successfully. `demo_eval.py --out-dir results` completed and wrote reports.

## 2) Complete workflow (sample query)
Query: `Why did global chip prices rise in 2025?` via `build_orchestrator(s).run(...)`
- Planner produced subtasks; researcher/critic ran; graph grew (nodes_added=10, edges_added=17).
- Embeddings + FAISS used (mock mode). 
- Relation detection created typed edges (including `CONTRADICTS`) recorded in results and graph.
- Retrieval surfaced contradictory evidence (contexts showed contradicting items).
- Synthesizer produced a cited final answer; `answer_grounded=True`, citations point to workspace units.

Evidence preserved: graph edges include contradictions (e.g. critic units linking to hypotheses with `CONTRADICTS`). Final answer cites real `iu_*` ids from retrieved context.

## 3) Streamlit app
- Entry point: `app.py` (`from latent_workspace.ui.app import main; if __name__=="__main__": main()`)
- Documented command: `streamlit run app.py` (and UI extra `.[ui]`).
- `WorkspaceSession` is Streamlit-free and testable; app imports Streamlit lazily with clear error if missing.
- UI renders: ask form, final answer, metrics, Run trace (subtasks/units/edges), Retrieved contexts (per-context + prompt block), History, and Workspace (Nodes/Edges). Sidebar save/load/reset.
- Smoke import: `import app; from latent_workspace.ui import WorkspaceSession` OK.
- No runtime integration errors observed in code paths.

## 4) Persistence and configuration
- Graph save/load: tested roundtrip (nodes/edges preserved). Snapshots write `graph.json` + `graph.faiss` (atomic write for json).
- Mock mode: `mock_mode=True` by default; `build_llm/build_embedder/build_relation_detector` use mock implementations. No API keys/network required.
- Live config: `Settings` reads env vars with prefix `DLRW_` (pydantic-settings), `.env` supported. Verified `DLRW_MOCK_MODE=false`, `DLRW_LLM_API_KEY`, `DLRW_LLM_MODEL` propagate correctly.
- Secrets: `.gitignore` contains `.env`, `.venv/`, `venv/`. No secrets committed.

## 5) Evaluation audit
Files inspected: `results/evaluation_report.json`, `results/evaluation_report.md`

### Formula (from code)
`composite = 0.35*retrieval_recall + 0.15*retrieval_precision + 0.20*conflict_component + 0.20*grounded_key_point_recall + 0.10*citation_precision`, rounded to 4 decimals.  
`conflict_component = 1.0` if no conflict units in case, else `conflict_recall`.  
All metrics in [0,1] computed by deterministic substring/ID matching (no judge model).

### Metric definitions
- retrieval_recall: fraction of relevant units (per case markers) present in retrieved contents
- retrieval_precision: fraction of retrieved contents that are relevant to the case
- conflict_recall: fraction of conflict units retrieved (or None/treated as 1.0 if none)
- citation_precision: fraction of citations in answer that are valid workspace unit ids
- citation_recall: fraction of relevant units that were cited
- answer_key_point_recall: fraction of case key points found in the answer text
- grounded_key_point_recall: fraction of case key points found in (answer + cited units)
- composite: weighted sum above

### Reported numbers (from JSON)
| system | composite | retrieval_recall | retrieval_precision | conflict_recall | citation_precision | citation_recall | answer_key_point_recall | grounded_key_point_recall | context tok (mean) | total tok (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| A: llm_only | 0.0670 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2500 | 0.3333 | 11 | 238 |
| B: flat_context | 0.9533 | 1.0000 | 0.8000 | 1.0000 | 1.0000 | 1.0000 | 0.8334 | 0.9167 | 198 | 388 |
| C: vector_rag | 0.7740 | 0.6875 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 0.5833 | 0.6667 | 112 | 318 |
| D: drlw | 0.7906 | 0.6875 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3333 | 0.5000 | 4461 | 4871 |
| ablation: no_expansion | 0.7187 | 0.6250 | 1.0000 | 0.7500 | 1.0000 | 1.0000 | 0.3333 | 0.5000 | 3974 | 4610 |

### Baselines comparability
All systems run over the same `DEFAULT_CASES` (4 cases, same units/markers/key points), same `Settings` (mock), same deterministic id handling in harness. Tasks/settings are comparable.

### Ablation
`NoExpansionSystem` uses `SubgraphRetriever` with `expansion_depth=0` but same orchestration/wiring as `OrchestratorSystem`. It changes only graph expansion; confirmed by code. Report shows D > ablation on composite/retrieval/conflict_recall.

### Flat-context baseline
Score 0.9533 > DRLW 0.7906 is reported honestly in README and reports. B has full context (all units) so recall is perfect; it is a full-context upper bound (not scalable) and the docs now state this explicitly.

### Notes
- Results are measured outputs from the mock harness (created_at 2026-10-27T10:43:17.606505+00:00). No fabricated numbers. No claims of statistical significance (single run, deterministic but small sample of cases/metrics).

## Fixes made in this audit
- None required. All verification passed as-is; added audit doc only.

## Remaining unverified
- Live LLM integration (requires real API keys/network) not executed; code paths and env wiring verified.
- Streamlit UI runtime not launched interactively in this session (imports/start command verified).
- Large-scale/statistical evaluation not performed.
```