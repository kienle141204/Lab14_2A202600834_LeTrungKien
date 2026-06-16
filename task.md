# Task List — Day 14: RAG Evaluation & Benchmarking

Tổng hợp tất cả việc cần làm trong lab này, theo thứ tự nên làm. Đánh dấu `[x]` khi xong.

> **Workflow quan trọng:** Code trong `template.py`, chạy `pytest tests/ -v`, khi pass hết thì
> **copy `template.py` → `solution/solution.py`** (test ưu tiên đọc `solution/solution.py`,
> fallback về `template.py`). KHÔNG đổi signature class/hàm — test khởi tạo object theo
> **thứ tự field (positional)**.

---

## PHẦN A — Code (`template.py`) — chiếm 50 điểm (pass pytest)

### Task 1 — Data Models
- [ ] `QAPair` (dataclass) — định nghĩa field đúng thứ tự: `question, expected_answer, context, metadata`
  - Ghi chú: `context: str = ""`, `metadata: dict = field(default_factory=dict)`,
    `retrieved_contexts: list = field(default_factory=list)`.
  - ⚠️ Test gọi `QAPair("q", "expected", None, {})` → `context` có thể là `None`, đừng giả định luôn là str.
- [ ] `EvalResult` (dataclass) — field theo thứ tự: `qa_pair, actual_answer, faithfulness, relevance, completeness, passed, failure_type` + optional `context_precision`, `context_recall` (mặc định `None`).
- [ ] `EvalResult.overall_score()` — trả về trung bình **3 metric answer-side**: `(faithfulness + relevance + completeness) / 3`. KHÔNG tính context_recall/precision.

### Task 2 — RAGASEvaluator (answer-side, word-overlap heuristic)
- [ ] `evaluate_faithfulness(answer, context)` → `|answer ∩ context| / |answer|`. Clamp `[0,1]`. Answer rỗng → `1.0`.
- [ ] `evaluate_relevance(answer, question)` → `|answer ∩ question| / |question|`. Clamp. Question rỗng → `1.0`.
- [ ] `evaluate_completeness(answer, expected)` → `|answer ∩ expected| / |expected|`. Clamp. Expected rỗng → `1.0`.
  - Ghi chú: dùng `_tokenize()` có sẵn (lowercase + bỏ STOPWORDS).
- [ ] `run_full_eval(answer, question, context, expected)` → tạo `EvalResult`.
  - `passed = True` nếu cả 3 score ≥ 0.5.
  - `failure_type` (first match wins): faithfulness < 0.3 → `"hallucination"`; relevance < 0.3 → `"irrelevant"`; completeness < 0.3 → `"incomplete"`; còn lại nếu fail → `"off_topic"`.

### Task 2b — RAGASEvaluator (retrieval-side, chạy trên `list[str]` chunks)
- [ ] `evaluate_context_recall(contexts, expected)` → `|expected ∩ (⋃ chunks)| / |expected|`. Clamp. Expected rỗng → `1.0`.
- [ ] `evaluate_context_precision(contexts, expected, relevance_threshold=0.1)` → **rank-aware AP@K**.
  - Chunk "relevant" nếu `|chunk ∩ expected| / |expected| >= threshold`.
  - `Precision@k = (#relevant trong top-k) / k`; `AP@K = (1/#relevant) · Σ_k [Precision@k · relevant_k]`.
  - Expected rỗng → `1.0`; không chunk / không relevant → `0.0`.
  - ⚠️ Đổi thứ tự chunk PHẢI thay đổi điểm (relevant lên đầu → điểm cao hơn).
- [ ] `rerank_by_overlap(contexts, query)` (hàm module-level) → sort chunk theo overlap với query, nhiều nhất lên đầu. Không được làm precision GIẢM.

### Task 3 — LLMJudge
- [ ] `__init__(self, judge_llm_fn)` → lưu `judge_llm_fn`.
- [ ] `score_response(question, answer, rubric)` → build prompt (gồm question + answer + rubric), gọi `judge_llm_fn`, parse JSON scores. Parse lỗi → mặc định `0.5` mỗi tiêu chí. Trả về `{"scores": {...}, "reasoning": str}`.
- [ ] `detect_bias(scores_batch)` → trả `{"positional_bias", "leniency_bias", "severity_bias"}` (bool).
  - leniency: avg > 0.8; severity: avg < 0.3; positional: response đầu điểm cao hơn đều đặn.

### Task 4 — BenchmarkRunner
- [ ] `run(qa_pairs, agent_fn, evaluator)` → mỗi pair gọi `agent_fn(pair.question)` rồi `evaluator.run_full_eval(...)`. Trả `list[EvalResult]` đúng số lượng input.
- [ ] `generate_report(results)` → `{total, passed, pass_rate, avg_faithfulness, avg_relevance, avg_completeness, failure_types}`.
- [ ] `run_regression(new_results, baseline_results)` → tính avg từng metric, so sánh; metric tụt **> 0.05** = regression. Trả các key `new_avg_*`, `baseline_avg_*`, `regressions: list[str]`, `passed: bool`.
  - ⚠️ Tên metric trong `regressions` dùng đúng `"faithfulness"`, `"relevance"`, `"completeness"` (test check `in`).
- [ ] `identify_failures(results, threshold=0.5)` → lọc result có bất kỳ score < threshold.

### Task 5 — FailureAnalyzer
- [ ] `categorize_failures(failures)` → đếm theo `failure_type`. Trả `dict[str,int]`.
- [ ] `find_root_cause(failure)` → so score thấp nhất, trả về đúng 1 trong 4 chuỗi cố định trong docstring (test có thể check substring → giữ nguyên wording).
- [ ] `generate_improvement_suggestions(failures)` → ≥ 3 chuỗi actionable (ít hơn nếu failures rỗng).
- [ ] `generate_improvement_log(failures, suggestions)` → bảng Markdown `| Failure ID | Type | Root Cause | Suggested Fix | Status |`. Status luôn `"Open"`.

### Verify code
- [x] `pytest tests/ -v` — pass toàn bộ (**39/39 PASS**).
- [x] `python template.py` — chạy được `__main__` block (smoke test với mock agent).
- [x] Copy `template.py` → `solution/solution.py`.

> ✅ Tất cả Task 1–5 đã implement xong và pass test (50đ code đã chốt).

---

## PHẦN B — `exercises.md` — chiếm ~25 điểm (dataset + rubric + reranking)

### Part 1 — Warm-up (lý thuyết)
- [ ] Ex 1.1 — Bảng RAGAS metric thresholds (acceptable vs critical low score + action) cho 5 metric.
- [ ] Ex 1.2 — Position bias: thiết kế experiment, fix verbosity bias, vì sao calibrate against human.
- [ ] Ex 1.3 — CI/CD thresholds cho mỗi metric + khi nào offline vs online eval.

### Part 3 — Extended
- [ ] Ex 3.1 — **Golden dataset 20 QA pairs** stratified: 5 Easy + 7 Medium + 5 Hard + 3 Adversarial. (Dùng domain từ Day 2.)
- [ ] Ex 3.2 — Chạy `BenchmarkRunner` trên 20 QA, điền bảng kết quả + aggregate report + 3 câu điểm thấp nhất.
- [ ] Ex 3.3 — Thiết kế rubric LLM-as-Judge 1–5 (domain-specific) + chọn 3–5 criteria dimensions + 3 edge cases khó score.
- [ ] Ex 3.4 (Bonus +10) — So sánh 2 framework (RAGAS/DeepEval/TruLens) trên cùng dataset.
- [ ] Ex 3.5 — **Reranking & Context Precision**: đo Recall + Precision baseline → rerank → đo lại → trả lời 4 câu phân tích + bảng kỹ thuật get-context (chọn ≥3) + (tuỳ chọn) viết reranker riêng.

---

## PHẦN C — `reflection.md` — chiếm ~15 điểm (failure analysis)

- [ ] Mục 1 — Benchmark results summary (pass rate, avg/min/max/std, score interpretation, failure distribution).
- [ ] Mục 2 — **Top 3 worst failures + 5 Whys analysis** mỗi cái (symptom → why 1–4 → root cause + proposed fix).
- [ ] Mục 3 — Failure clustering (gom theo root cause, chọn priority, chọn 1 cluster để fix).
- [ ] Mục 4 — Paste output `generate_improvement_log()` + 3 suggestions.
- [ ] Mục 5 — Regression testing strategy (CI/CD integration point, threshold 0.05, block vs alert, vị trí eval trong flow).
- [ ] Mục 6 — Continuous improvement loop (3 actions tiếp theo + failure cases mới thêm vào benchmark).
- [ ] Mục 7 — Framework reflection (chọn framework cho production + lý do).

---

## Checklist nộp bài (gốc từ README + exercises.md)
- [ ] `pytest tests/ -v` pass hết
- [ ] `overall_score`, `run_regression`, `generate_improvement_log` đã implement
- [ ] `evaluate_context_recall` + `evaluate_context_precision` (Task 2b) đã implement
- [ ] Ex 3.5 hoàn thành (đo Recall/Precision + reranking before/after)
- [ ] `exercises.md`: golden dataset 20 QA + benchmark results + rubric
- [ ] `reflection.md`: 3 failures (5 Whys) + improvement log + CI/CD strategy
- [ ] `solution/solution.py` đã copy

## Bonus (thêm điểm)
- [x] +10 — Chạy RAGAS THẬT vs heuristic trên cùng 20 QA (`bonus/compare_ragas.py`).
      → Avg relevance 0.250 (heur) vs 0.752 (RAGAS); kết quả ở Ex 3.4 + reflection §7.
- [x] +5 — Tích hợp eval vào CI/CD script (`.github/workflows/eval.yml` + `ci_gate.py`, gate PASS local).
- [x] +5 — Thêm custom metric `evaluate_conciseness` (anti-verbosity, test vẫn 39/39).

## Trạng thái tổng (cập nhật)
- ✅ Code 50đ — 39/39 pytest PASS, `solution/solution.py` đã copy.
- ✅ `exercises.md` — golden dataset 20 QA + benchmark thật + rubric + reranking (số liệu thật).
- ✅ `reflection.md` — 3 failures 5-Whys + clustering + improvement log + CI/CD + framework reflection.
- ✅ Bonus +20 (custom metric + CI/CD + RAGAS thật).
- 🎯 Ước tính: 100/100 + 20 bonus.
