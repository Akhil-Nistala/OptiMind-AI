# Execution log — 28 October 2026

## Phase 1 — Environment and small datasets

Created requirements.txt, configs/models.json, configs/experiments.json, database/schema.sql, scripts/prepare_data.py and scripts/download_assets.py. Inspected installed Python packages; downloaded missing OR-Tools, LangGraph, Transformers, BM25 and PDF dependencies and extracted wheels into a workspace-only environment. Downloaded real pinned MiniLM and FLAN-T5 safetensors with provenance hashes. No API keys, paid calls or global package changes were needed. Created 42 explicitly synthetic records using seed 42, independent reference totals, four authored scenario text procedures, and two real public DOE PDFs. An NLR PDF host returned HTTP 403; the same DOE tip sheet was obtained from energy.gov. Source notices are retained, a third-party table is excluded from ingestion, and original PDFs are excluded from the archive.

Commands: dependency/import inspection; pip download; `python -m scripts.download_assets --only models`; `python -m scripts.download_assets --only documents`; `python -m scripts.prepare_data`. Next: run the smallest solver and retrieval paths.

## Phase 2 — Scheduling

Created optimization/problem.py, greedy_baseline.py, solver.py, visualization.py and benchmark.py. Implemented integer starts/ends, precedence, exclusive machines, maintenance blackouts, release/availability constraints, optional hard deadlines, shared crew demand and makespan objective. Implemented an independent schedule validator. Initial 13-instance run: mean makespan reduction 12.69%, zero violations, 6 solver optima proved. Added standard FT06 for a 14-instance final benchmark: FT06 greedy 88 versus proved optimum 55. Final metrics and all bound/proof statuses are saved in scheduling/benchmark.csv and summary.json. Hard wall-time limits explain possible incumbent/runtime variation.

Commands: `python -m optimization.benchmark`. Tests include an exhaustive independently enumerated optimum, public-example optimum 11, capacity/calendars/releases, infeasibility, UNKNOWN, heuristic failure and tampered schedules. No unproved incumbent is called optimal. Next: ingest documents and compare retrieval methods.

## Phase 3 — Ingestion and retrieval

Created knowledge_assistant/ingestion.py and retrieval.py. PDF page provenance, text document metadata and SHA256 source hashes are preserved. Initial 37 chunks became 36 after excluding a third-party table and repairing PDF line-end hyphenation. Built real MiniLM 384-dimensional normalized embeddings, BM25 and hybrid reciprocal-rank fusion. Index manifests validate the vector/chunk ordering. Gold relevance is document-level; no independent human annotation is claimed.

Commands: `python -m knowledge_assistant.ingestion`; `python -m knowledge_assistant.retrieval --build`. Final Recall@3 and MRR@5 are saved in knowledge/summary.json. Next: verify local model generation and citation handling.

## Phase 4 — Grounded answers and evaluation

Created generation.py, evaluation.py and experiments/retrieval_questions.json. Ran actual local FLAN-T5 inference on 18 answerable and four unsupported authored questions. The initial pipeline obtained 10/18 answerable keyword checks and is preserved in knowledge_initial. Added sentence-level reranking and stricter exact-span draft validation, with explicitly labeled extractive fallback. Final answerable keyword checks: 15/18; all four unsupported queries abstained; citation source/span checks passed for all answered cases. Only three final answers accepted a model draft; 15 used the labeled evidence fallback. This is LLM-assisted retrieval with conservative validation, not a claim that small-model generation is generally reliable. Improvements were developed against this set, which is not a held-out test benchmark.

Commands: `python -m knowledge_assistant.evaluation`. Unit tests cover reproducible chunk IDs, metadata, fabricated model text rejection, exact-span acceptance, unknown equipment identifiers and pre-generation abstention. Next: integrate deterministic analysis.

## Phase 5 — Structured analysis

Created database/queries.py and operations_agent/data_analysis.py. Investigative SQL is fixed, parameterized, read-only and query-only. Pandas computes actual output totals, shift denominators, targets, utilization, planned/actual time and known downtime. Machine inventory coverage and contradictory values are checked. Independent Python reference totals from the original generator records verify the SQL/Pandas path. Scenario: 2101 to 1918 units (-183), known downtime 643 to 1236 minutes, one missing downtime record. Missing downtime is never reported as a complete total or silently imputed.

Tests: numerical references, write rejection, empty periods, invalid dates, contradictory shift totals and an entire machine missing from both periods. Next: coordinate tools using LangGraph.

## Phase 6 — Agent workflow

Created operations_agent/workflow.py and evaluation.py. Implemented Understand, Plan, Investigate, Validate, conditional Optimize and Report stages. The bounded routing policy selects only supported tools. Inputs/outputs, durations, errors and stage traces are retained. Schedules are explicit proposals with human review. Observations are calculated facts; explanations remain labeled hypotheses. No model-generated SQL, arbitrary command execution or plant-control connection exists.

The first evaluation detected that an empty period could be marked supported by generic reference material. Corrected support gating and added regression coverage. Final 12-case authored suite: exact tool selection 100%, numerical references 100%, support decisions 100%, schedule feasibility 100%, zero unnecessary calls under the expected tool policy. Structural grounding and citation-span checks pass; they are not independent semantic entailment reviews.

Command: `python -m operations_agent.evaluation`. Next: exercise interface behavior and finalize tests.

## Phase 7 — Interface and tests

Created app/streamlit_app.py, tests/ and pytest.ini. The app provides editable scheduling JSON, live solver comparison, technical question answering with source passages/model validation, and an operations report with explicit date fields and downloadable JSON. Streamlit AppTest runs actual scheduling, invalid-input, SQL-only and local-model paths. Initial suite: 25 passed in 5.32s. Expanded suite: 31 passed; the final XML test record is results/pytest.xml. One upstream LangChain pending-deprecation warning is retained; no test failed because of it.

Command: `python -m pytest -q --junitxml=results/pytest.xml`. Model-dependent UI integration is skipped in CI if assets are absent, but it ran locally. Next: package tested source and inspect figures.

## Phase 8 — Documentation and reproducibility

Created README.md, NOTICE.md, this log, REPORT.md, provenance artifacts, installed-dependency snapshot, GitHub Actions workflow and scripts/bootstrap.py, run_experiments.py and package_project.py. Python source was formatted with Black. Actual scheduling/RAG/agent figures and failure analysis are included. The portable archive contains source, reports, fixtures, prepared SQLite/index and evaluation outputs; it excludes original PDFs, large model weights, caches and Git metadata. Bootstrap retrieves the excluded assets from pinned sources. A local Git repository/commit is created; GitHub publication and remote CI execution are not claimed.

Remaining limitations: small development evaluations, synthetic production data, rule-bounded routing, three known answer keyword failures, extractive fallback dependence, PDF layout limitations, wall-time-sensitive solver incumbents, and no physical production execution or causal diagnosis. No optional cloud model, FAISS index or unmeasured performance claim was added.
