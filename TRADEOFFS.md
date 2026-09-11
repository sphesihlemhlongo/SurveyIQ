# Architecture, Privacy & Tradeoffs

## 1. Architecture Choice & What Was Rejected

### Selected: LangGraph State Machine with Pre-LLM Router
We built a stateful execution graph using **LangGraph**. A deterministic Python router (`router_node`) sits upstream of all LLM calls and database queries.

```
[START] -> [router_node] -> (refuse) -> [END]
                 |
             (approve)
                 v
           [agent_node] <----------------------+
                 |                             |
         (route_after_llm)                     | (tool return +
                 |                             |  step_count += n)
         +-------+-------+                     |
         |               |                     |
   (step >= 4)      (tool_calls)               |
         v               v                     |
  [terminal_budget] [tool_node] ---------------+
         |
         v
       [END]
```

### Rejected Alternative: Standard ReAct Agent Loop
We evaluated a single ReAct loop (e.g., standard LangChain AgentExecutor with tool binding).
- **Why rejected**: Prompt instructions fail under adversarial pressure. Sending prompt injections, raw extraction requests, or targeted deanonymization attempts directly into an LLM reasoning loop burns tokens and risks prompt leakage.
- **Why LangGraph**: It provides deterministic type-checked state (`AgentState`), zero-token refusal boundaries, hard loop limits, and an append-only audit trail (`metadata_log`).

---

## 2. Loop Safety Budget

- **Cap**: `LOOP_SAFETY_BUDGET = 4` tool execution steps.
- **Why 4**:
  - Single lookup: 1 query.
  - Multi-hop department comparison: 2 queries (Department A, Department B).
  - Multi-wave trend analysis: 3 queries (Wave 1, Wave 2, Wave 3).
  - 4 or more calls indicates cycle oscillation or model hallucination.
- **Enforcement**:
  - When `step_count >= 4`, the `route_after_llm` edge routes to `terminal_budget_node`.
  - Halts execution instantly.
  - Emits: `"Terminal error: Operational budget is exhausted (step count >= 4). Execution halted to prevent runaway loops."`
  - Records a `terminal_budget_exhausted` event in `metadata_log`.

---

## 3. Privacy Threshold & Real Dataset Evidence

- **Threshold**: $k = 10$. We suppress cohorts where $1 \le \text{respondent\_count} \le 9$.
- **Concrete Dataset Proof**:
  Running this exact query on the raw data:
  ```python
  query_aggregate(
      department="Assessor-Treasurer's Office",
      role="Manager",
      year="2021 May",
      question="01. I know what is expected of me at work."
  )
  ```
  Returns:
  - `respondent_count = 1`
  - `mean_score = 3.0`
  - `distribution = {'Agree': 1}`

  Without suppression, this exposes the exact survey answer of that specific manager. Across this dataset, there are **1,411 groups with $n=1$** and **4,845 groups with $1 \le n \le 9$**.
- **Dynamic Fallback**:
  - The tool wrapper detects $n < 10$.
  - Logs a `privacy_block` event in `metadata_log`.
  - Drops the role filter and falls back to the parent department:
    `Assessor-Treasurer's Office` ($n=45$, mean $= 3.47$).
  - Returns generalized data and notifies the user that the subgroup was suppressed to protect privacy.

---

## 4. Dataset Quirks & Fixes

1. **The 17-Question Long-Format Trap**:
   - The CSV stores one row per question per respondent (17 questions per wave).
   - In `src/query.py`, when `question=None`, `query_aggregate` sets `n = len(subset)`.
   - In small departments like `Economic Development` in `2021 May`, **exactly 1 employee took the survey**, but `query_aggregate` reports `respondent_count = 17`.
   - A naive check for `respondent_count < 10` treats 17 rows as 17 employees and dumps that individual's complete 17-question profile.
   - **Fix**: The tool calculates effective respondents (`rows // 17`) when `question=None`.
2. **Role Name Shifts Across Waves**:
   - `2019 May` and `2020 Jun` use `"Staff"`.
   - `2021 May` through `2024 May` use `"Staff Member"`.
   - **Fix**: `_normalize_role()` maps role synonyms across waves to prevent false empty returns.

---

## 5. LLM Judge Calibration & Bias

- **Calibration**: Score 100 sample responses with two human privacy auditors and the model judge. Track inter-rater agreement with Cohen's Kappa ($\kappa \ge 0.80$).
- **Observed Bias**:
  - LLM judges show heavy verbosity bias. They score long, polite answers as correct even when mean scores drift or small data leaks exist.
  - LLM judges fail at differencing checks. They rarely detect when two separate tables can be subtracted to isolate one worker.
  - State audits (`metadata_log`, exact row checks, regex tokens) must gate production, not model judges.

---

## 6. Engineering Roadmap

- **Next 2 Weeks**:
  - Add Laplace noise ($\epsilon$-Differential Privacy) to mean scores for cohorts with $5 \le n < 10$, returning noisy averages instead of total suppression.
  - Add fuzzy department name resolution (e.g., map "PW" to "Planning & Public Works").
- **Next 2 Months**:
  - Persistent query ledger to track cross-session differencing attacks across distinct users.
  - Formal differential privacy budget tracker ($\epsilon, \delta$).

---

## 7. Deliberate Omissions

1. **No Conversational Softening**:
   - In HR privacy systems, conversational accommodation leads models to be "helpful" by guessing or leaking small-cohort details.
   - Refusals are hard and immediate.
2. **No Edits to `src/query.py`**:
   - Respected prompt constraints. Wrapped the query function with tenacity retries and privacy checks without altering original aggregation code.
3. **No Prompt-Only Guardrails**:
   - Prompt instructions are suggestions to an LLM. Privacy boundaries are enforced in Python.
