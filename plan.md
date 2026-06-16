# Plan — Hoàn thiện Day 14: RAG Evaluation & Benchmarking

Kế hoạch thực thi đầy đủ để hoàn thành mọi task trong [task.md](task.md). Chia 5 giai đoạn
theo thứ tự phụ thuộc, kèm cách triển khai cụ thể cho từng hàm và lệnh verify.

> **Nguyên tắc xuyên suốt**
> - Code trong `template.py`, KHÔNG đổi signature, dùng `_tokenize()` có sẵn.
> - Clamp mọi metric về `[0,1]`; mẫu số = 0 (input rỗng) → trả `1.0` (riêng `context_precision` → `0.0` khi không chunk/không relevant).
> - Sau khi pass test → copy `template.py` sang `solution/solution.py`.
> - Verify liên tục bằng `pytest tests/ -v`.

---

## GIAI ĐOẠN 0 — Chuẩn bị (5 phút)

1. Chạy baseline để thấy test fail: `pytest tests/ -v` (kỳ vọng: lỗi `NotImplementedError`/`TypeError`).
2. Mở song song `template.py` + `tests/test_solution.py` để đối chiếu kỳ vọng từng hàm.
3. Helper dùng lại nhiều lần:
   ```python
   def _overlap_ratio(a: set, b: set) -> float:   # |a ∩ b| / |b|
       return len(a & b) / len(b) if b else 1.0
   ```
   (Không bắt buộc tạo hàm này, nhưng pattern lặp ở 5 metric.)

---

## GIAI ĐOẠN 1 — Data Models (Task 1)

**Mục tiêu test:** `TestEvalResultOverallScore`, và mọi test khác đều cần khởi tạo được object.

- `QAPair`:
  ```python
  @dataclass
  class QAPair:
      question: str
      expected_answer: str
      context: str = ""
      metadata: dict = field(default_factory=dict)
      retrieved_contexts: list = field(default_factory=list)
  ```
  ⚠️ Thứ tự field phải đúng — test gọi `QAPair("q", "expected", None, {})` (positional, `context=None`).
- `EvalResult`:
  ```python
  @dataclass
  class EvalResult:
      qa_pair: QAPair
      actual_answer: str
      faithfulness: float
      relevance: float
      completeness: float
      passed: bool
      failure_type: str | None = None
      context_precision: float | None = None
      context_recall: float | None = None
  ```
- `overall_score()`: `return (self.faithfulness + self.relevance + self.completeness) / 3.0`

**Verify:** `pytest tests/test_solution.py::TestEvalResultOverallScore -v`

---

## GIAI ĐOẠN 2 — RAGASEvaluator (Task 2 + 2b)

**Mục tiêu test:** `TestRAGASEvaluator`, `TestContextMetrics`.

### 2.1 Answer-side (3 hàm overlap)
- `evaluate_faithfulness(answer, context)`: `a=_tokenize(answer)`; nếu `not a` → `1.0`; else `len(a & _tokenize(context)) / len(a)`.
- `evaluate_relevance(answer, question)`: mẫu số là `_tokenize(question)`; rỗng → `1.0`.
- `evaluate_completeness(answer, expected)`: mẫu số là `_tokenize(expected)`; rỗng → `1.0`.
- Tất cả `min(1.0, max(0.0, score))`.

### 2.2 Retrieval-side
- `evaluate_context_recall(contexts, expected)`:
  ```python
  exp = _tokenize(expected)
  if not exp: return 1.0
  union = set().union(*[_tokenize(c) for c in contexts]) if contexts else set()
  return min(1.0, len(exp & union) / len(exp))
  ```
- `evaluate_context_precision(contexts, expected, relevance_threshold=0.1)`:
  ```python
  exp = _tokenize(expected)
  if not exp: return 1.0
  if not contexts: return 0.0
  relevant_flags = [len(_tokenize(c) & exp) / len(exp) >= relevance_threshold for c in contexts]
  num_rel = sum(relevant_flags)
  if num_rel == 0: return 0.0
  hits = 0; ap = 0.0
  for k, is_rel in enumerate(relevant_flags, start=1):
      if is_rel:
          hits += 1
          ap += (hits / k)        # Precision@k tại vị trí relevant
  return ap / num_rel
  ```
  ⚠️ Đây là điểm dễ sai nhất: phải **rank-aware** thì test `test_context_precision_rewards_relevant_first` mới pass (relevant đứng trước → điểm cao hơn).

### 2.3 `run_full_eval(answer, question, context, expected)`
- Tính 3 score; `passed = all(s >= 0.5)`.
- `failure_type` (first match wins): `faithfulness < 0.3 → "hallucination"` → `relevance < 0.3 → "irrelevant"` → `completeness < 0.3 → "incomplete"` → else nếu `not passed` → `"off_topic"`, ngược lại `None`.
- Trả `EvalResult(qa_pair=QAPair(question, expected, context), actual_answer=answer, ...)`.

### 2.4 `rerank_by_overlap(contexts, query)` (module-level, cho Ex 3.5)
```python
return sorted(contexts, key=lambda c: len(_tokenize(c) & _tokenize(query)), reverse=True)
```
⚠️ `sorted` ổn định → không làm precision giảm (test `test_reranking_improves_or_keeps_precision`).

**Verify:** `pytest tests/test_solution.py::TestRAGASEvaluator tests/test_solution.py::TestContextMetrics -v`

---

## GIAI ĐOẠN 3 — LLMJudge (Task 3)

**Mục tiêu test:** `TestLLMJudge` (mock trả `'{"accuracy": 0.8, "clarity": 0.7}'`).

- `__init__`: `self.judge_llm_fn = judge_llm_fn`.
- `score_response(question, answer, rubric)`:
  1. Build prompt gồm question + answer + từng `criterion: description` + yêu cầu trả JSON.
  2. `raw = self.judge_llm_fn(prompt)`.
  3. `try: scores = json.loads(raw)` (lọc về float, clamp `[0,1]`); `except: scores = {c: 0.5 for c in rubric}`.
  4. `return {"scores": scores, "reasoning": raw}`.
  - Nhớ `import json`.
- `detect_bias(scores_batch)`:
  - Gom mọi giá trị score → `avg`.
  - `leniency_bias = avg > 0.8`; `severity_bias = avg < 0.3`.
  - `positional_bias`: so điểm trung bình của phần tử đầu vs còn lại (đơn giản: `False` nếu batch < 2; hoặc kiểm tra phần tử đầu cao hơn rõ rệt). Test chỉ check key tồn tại → trả bool hợp lý.
  - `return {"positional_bias":..., "leniency_bias":..., "severity_bias":...}`.

**Verify:** `pytest tests/test_solution.py::TestLLMJudge -v`

---

## GIAI ĐOẠN 4 — BenchmarkRunner + FailureAnalyzer (Task 4 + 5)

**Mục tiêu test:** `TestBenchmarkRunner`, `TestRunRegression`, `TestFailureAnalyzer`, `TestGenerateImprovementLog`.

### 4.1 BenchmarkRunner
- `run`: `return [evaluator.run_full_eval(agent_fn(p.question), p.question, p.context, p.expected_answer) for p in qa_pairs]`.
- `generate_report`: tính `total, passed, pass_rate`, ba `avg_*` (cẩn thận chia 0 khi rỗng), `failure_types` đếm bằng dict.
- `run_regression(new, baseline)`:
  - Hàm phụ `avg(results, attr)`.
  - Với mỗi metric trong `["faithfulness","relevance","completeness"]`: nếu `baseline_avg - new_avg > 0.05` → thêm tên metric vào `regressions`.
  - Trả `new_avg_*`, `baseline_avg_*`, `regressions`, `passed = len(regressions)==0`.
  - ⚠️ Tên key/metric phải đúng chính tả (test check `'faithfulness' in result['regressions']`).
- `identify_failures(results, threshold=0.5)`: lọc result có `min(faithfulness, relevance, completeness) < threshold`.

### 4.2 FailureAnalyzer
- `categorize_failures`: đếm theo `failure_type` (bỏ qua `None`). Trả dict (rỗng vẫn là dict).
- `find_root_cause(failure)`: tìm score thấp nhất trong 3 → map sang đúng chuỗi cố định trong docstring:
  - faithfulness thấp nhất → `"Context is missing or irrelevant — improve retrieval"`
  - relevance thấp nhất → `"Answer does not address the question — improve prompt clarity"`
  - completeness thấp nhất → `"Answer is missing key information — increase context window or improve generation"`
  - nhiều score cùng thấp/đều thấp → `"Multiple issues detected — review full pipeline"`
  - ⚠️ Giữ nguyên wording (có thể bị check substring).
- `generate_improvement_suggestions(failures)`: dựa trên `categorize_failures`, sinh ≥ 3 chuỗi actionable; `failures` rỗng → trả ít hơn (hoặc `[]`).
- `generate_improvement_log(failures, suggestions)`:
  ```
  | Failure ID | Type | Root Cause | Suggested Fix | Status |
  |------------|------|------------|---------------|--------|
  | F001 | <failure_type> | <find_root_cause> | <suggestions[i] nếu có> | Open |
  ```
  - `Status` luôn `"Open"`; suggestions ngắn hơn failures thì để trống ô fix.

**Verify:** `pytest tests/test_solution.py::TestBenchmarkRunner tests/test_solution.py::TestRunRegression tests/test_solution.py::TestFailureAnalyzer tests/test_solution.py::TestGenerateImprovementLog -v`

### Chốt code
- [ ] `pytest tests/ -v` xanh toàn bộ.
- [ ] `python template.py` chạy được `__main__`.
- [ ] Copy `template.py` → `solution/solution.py` (test sẽ tự chuyển sang đọc file này).

---

## GIAI ĐOẠN 5 — Tài liệu (exercises.md + reflection.md)

Làm sau khi code xong để dùng số liệu thật từ benchmark.

### 5.1 exercises.md
1. **Part 1 (lý thuyết)** — điền 3 bảng warm-up (thresholds, position bias, CI/CD thresholds).
2. **Ex 3.1 — Golden dataset 20 QA** (5E/7M/5H/3A) theo domain Day 2. Đây là phần tốn thời gian nhất → làm trong 1 đoạn tập trung.
3. **Ex 3.2 — Benchmark run**: viết script nhỏ (hoặc mở rộng `__main__`) chạy 20 QA → copy số liệu vào bảng + 3 câu điểm thấp nhất.
4. **Ex 3.3 — Rubric 1–5** domain-specific + 3–5 dimensions + 3 edge cases.
5. **Ex 3.5 — Reranking**: dùng dataset R01–R05 cho sẵn → đo Recall/Precision baseline → `rerank_by_overlap` → đo lại → trả lời 4 câu + bảng kỹ thuật.
6. (Bonus) Ex 3.4 — so sánh 2 framework.

### 5.2 reflection.md
1. Mục 1 — paste benchmark summary từ Ex 3.2.
2. Mục 2 — **3 worst failures + 5 Whys** (lấy từ `identify_failures` + `find_root_cause`).
3. Mục 3 — failure clustering theo root cause.
4. Mục 4 — paste `generate_improvement_log()` + 3 suggestions.
5. Mục 5–7 — regression strategy, continuous improvement loop, framework reflection.

---

## GIAI ĐOẠN 6 (tuỳ chọn) — Bonus điểm
- [ ] +10: chạy thêm 1 framework thật (RAGAS/DeepEval) trên cùng dataset, so điểm.
- [ ] +5: viết `.github/workflows/eval.yml` chạy pytest + threshold gate.
- [ ] +5: thêm 1 custom metric (vd: citation accuracy, length-penalty) vào `RAGASEvaluator`.

---

## Thứ tự ưu tiên khuyến nghị
1. GĐ 1→4 (code) — lấy 50đ pytest, nền tảng cho mọi thứ.
2. GĐ 5.1 Ex 3.1 + 3.2 — dataset & benchmark (25đ).
3. GĐ 5.2 — reflection 5 Whys (15đ).
4. GĐ 5.1 Ex 3.3 + 3.5 — rubric & reranking.
5. GĐ 6 — bonus nếu còn thời gian.

## Rủi ro / điểm dễ sai (đọc trước khi code)
- `context_precision` phải rank-aware (AP@K), không phải precision tổng → test so sánh thứ tự sẽ fail nếu làm sai.
- Field positional + `context=None` trong test → đừng `.lower()` trực tiếp lên context khi nó `None` (dùng `_tokenize` đã guard `if not text`).
- Tên metric trong `run_regression` và wording trong `find_root_cause` bị check substring → giữ chính xác.
- `overall_score` chỉ trung bình 3 metric answer-side, KHÔNG gồm context metrics.
