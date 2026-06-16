# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-day teaching lab (Day 14 of the AICB-P1 program) for building an **AI evaluation & benchmarking pipeline** with RAGAS-inspired metrics, LLM-as-Judge scoring, and failure analysis. The deliverable is a completed `solution/solution.py` plus the written `exercises.md` and `reflection.md`. The full task spec and theory (in Vietnamese) live in `README.md`.

## Commands

```bash
pytest tests/ -v            # run the full suite
pytest tests/test_solution.py::TestRAGASEvaluator -v          # one test class
pytest tests/test_solution.py::TestRAGASEvaluator::test_faithfulness_fully_grounded -v   # one test
python template.py          # manual smoke run via the __main__ block (mock agent)
```

No build step, no dependencies beyond `pytest` (and the stdlib). Run all commands from the repo root.

## How the test harness resolves the module under test

`tests/test_solution.py` dynamically imports, in priority order: `solution/solution.py` → `solution/app.py` → `template.py`. So:
- Until `solution/solution.py` exists, tests run against `template.py` and will fail (every method raises `NotImplementedError`).
- **The intended workflow is to implement everything in `template.py`, then copy it to `solution/solution.py`.** Keep the two in sync — tests prefer `solution/solution.py` once present.
- Never change class or function signatures; the tests construct objects positionally (see below).

## Architecture

The whole pipeline is one module (`template.py`) of five cooperating pieces. The data flows: `QAPair` → agent → `RAGASEvaluator.run_full_eval` → `EvalResult` → aggregated by `BenchmarkRunner` → diagnosed by `FailureAnalyzer`.

- **`QAPair`** (dataclass) — golden-dataset item. Field order matters for tests: `question, expected_answer, context, metadata`, plus `retrieved_contexts: list`. Tests pass `context=None` positionally, so don't assume it's always a string.
- **`EvalResult`** (dataclass) — per-question result. Field order: `qa_pair, actual_answer, faithfulness, relevance, completeness, passed, failure_type`, plus optional `context_precision` / `context_recall`. `overall_score()` is the mean of the **three answer-side** metrics only (faithfulness, relevance, completeness) — retrieval metrics are excluded.
- **`RAGASEvaluator`** — all metrics are **word-overlap heuristics**, not real LLM/embedding calls. Tokenization goes through the module-level `_tokenize()`, which lowercases, splits on `\b\w+\b`, and drops `STOPWORDS`. Two metric families:
  - Answer-side: `evaluate_faithfulness` (answer∩context / answer), `evaluate_relevance` (answer∩question / question), `evaluate_completeness` (answer∩expected / expected).
  - Retrieval-side (Task 2b): `evaluate_context_recall` operates on the **union** of a `list[str]` of chunks; `evaluate_context_precision` is **rank-aware Average Precision (AP@K)** — order of chunks changes the score. `rerank_by_overlap()` (module-level helper) reorders chunks by lexical overlap and must not *lower* precision.
- **`LLMJudge`** — wraps an injected `judge_llm_fn: Callable[[str], str]`. `score_response` builds a rubric prompt, calls the fn, parses JSON scores, and falls back to `0.5` per criterion on parse failure. `detect_bias` flags positional / leniency (avg > 0.8) / severity (avg < 0.3) bias across a batch.
- **`BenchmarkRunner`** — `run` executes agent + evaluator over all pairs; `generate_report` aggregates (pass rate, avg metrics, failure-type counts); `run_regression` compares averages vs a baseline and flags any metric that drops **> 0.05**; `identify_failures` filters results below a threshold.
- **`FailureAnalyzer`** — `categorize_failures` (count by type), `find_root_cause` (maps lowest score → fixed diagnostic string), `generate_improvement_suggestions` (≥3 actionable strings), `generate_improvement_log` (Markdown table, every row `Status = Open`).

## Conventions specific to this lab

- Clamp every metric to `[0.0, 1.0]`; return `1.0` for an empty denominator (empty answer/question/expected); `evaluate_context_precision` returns `0.0` for empty/no-relevant chunks.
- `run_full_eval` failure-type precedence (first match wins): faithfulness < 0.3 → `hallucination`, relevance < 0.3 → `irrelevant`, completeness < 0.3 → `incomplete`, else if failed → `off_topic`. `passed` is true when all three answer-side scores ≥ 0.5.
- `find_root_cause` and `generate_improvement_log` are asserted against by `in`/substring checks — keep their exact wording (e.g. the literal `"Open"` status, the documented root-cause sentences).
