# Architecture, Privacy & Tradeoffs Document

## 1. Architectural Justification & Rejected Alternatives

### Selected Architecture: LangGraph Deterministic State Machine with Dedicated Pre-Router
We chose a stateful directed acyclic/cyclic graph built on **LangGraph** with a dedicated, zero-latency **`router_node`** upstream of all model and database execution. The workflow operates as follows:
```
[START] -> [router_node] -> (if refused) -> [END]
                     |
              (if approved)
                     v
               [agent_node] <----------------------+
                     |                             |
     (route_after_llm: step_count >= 4)            | (tool return +
                     |                             |  step_count += 1)
     +---------------+---------------+             |
     | (>= 4 steps)  | (tool_calls)  | (no tools)  |
     v               v               v             |
 [terminal_budget] [tool_node] ----> [agent_node] -+
     |                               |
     v                               v
   [END]                           [END]
```

### Alternative Considered and Rejected: Single Unconstrained ReAct Agent
We evaluated and rejected a conventional monolithic ReAct agent (e.g. standard LangChain AgentExecutor with tool binding).
- **Why Rejected**: In a ReAct architecture, every input—including prompt injections, malicious deanonymization attempts, and administrative impersonations—is fed directly to the model's reasoning loop. Prompt-based guardrails ("Do not answer questions about individuals") are non-deterministic and vulnerable to jailbreaks.
- **Why LangGraph Wins**: LangGraph provides typed state (`AgentState`), deterministic routing boundaries that reject attacks in 0ms without consuming tokens, deterministic loop safety budgets, and an immutable audit trail (`metadata_log`) tracking privacy blocks and retries.

---

## 2. Loop-Safety Threshold & Boundary Behavior

- **Threshold Selection**: `LOOP_SAFETY_BUDGET = 4` steps.
- **Justification**:
  - A standard aggregate lookup requires **1 tool call** (1 step).
  - A multi-hop comparative query (e.g. comparing Human Services vs District Court) requires **2 tool calls** (2 steps).
  - A multi-wave comparative query requires **3 tool calls** (3 steps).
  - Any trajectory requiring 4 or more tool calls is either stuck in an oscillation loop (e.g. repeatedly querying invalid parameters) or suffering from reasoning drift.
- **Boundary Behavior**:
  If `step_count >= 4`, the conditional edge `route_after_llm` transitions to `terminal_budget_node`. It halts execution immediately, logs a `terminal_budget_exhausted` event in `metadata_log`, and emits the terminal error message:
  `"Terminal error: Operational budget is exhausted (step count >= 4). Execution halted to prevent runaway loops."`

---

## 3. $k$-Anonymity Threshold & Empirical Dataset Proof

- **Threshold**: $k = 10$. Any cohort where $1 \le \text{respondent\_count} \le 9$ is strictly suppressed.
- **Specific Query Case from this Dataset**:
  ```python
  query_aggregate(
      department="Assessor-Treasurer's Office",
      role="Manager",
      year="2021 May",
      question="01. I know what is expected of me at work."
  )
  ```
  - **Returned Result**:
    - `respondent_count = 1`
    - `mean_score = 3.0`
    - `distribution = {'Agree': 1}`
  - **The Risk**: Without suppression, this reveals the exact rating of the single individual manager in that department. Across the dataset, there are **1,411 groups** of size 1 and **4,845 groups** where $1 \le n \le 9$.
- **Dynamic Fallback Execution**:
  Instead of failing or returning blank data, our wrapper intercepts the $n=1$ violation, logs a `privacy_block` in `metadata_log`, strips the most specific filter (`role`), and queries the broader department level (`Assessor-Treasurer's Office`, $n=45$, mean $= 3.47$). The user receives a reliable aggregate accompanied by an explicit disclosure that the subgroup was suppressed for confidentiality.

---

## 4. Crucial Dataset Findings & Architectural Impact

1. **The 17-Question Long-Format Trap**:
   - The survey dataset is stored in long format: 1 row per question per respondent (17 questions per survey wave).
   - In `src/query.py`, when `question=None`, `query_aggregate` computes `n = len(subset)`.
   - In small departments (e.g., `Economic Development`, `2021 May`), **exactly 1 employee took the survey**, but `query_aggregate` returns `respondent_count = 17`!
   - A naive implementation checking `respondent_count < 10` mistakenly assumes 17 employees took the survey and leaks that single individual's complete 17-question profile.
   - **Our Fix**: Our tool detects effective respondent counts (`rows // 17`) when `question=None`, suppressing small cohorts regardless of whether the question was filtered.
2. **Role Naming Shift Across Survey Waves**:
   - In `2019 May` and `2020 Jun`, staff are categorized as `"Staff"`.
   - In `2021 May` through `2024 May`, staff are categorized as `"Staff Member"`.
   - **Our Fix**: `_normalize_role()` automatically maps role synonyms across survey waves so queries do not return false zeroes.

---

## 5. LLM-as-a-Judge: Human Agreement & Bias Direction

- **Checking Agreement**: Run a double-blind calibration sample of 100 test query responses scored by human privacy auditors vs the LLM judge. Calculate inter-rater reliability using **Cohen's Kappa** ($\kappa \ge 0.80$) and evaluate confusion matrices specifically on privacy leak edge cases.
- **Direction of Bias**:
  - LLM judges exhibit **sycophancy and verbosity bias**: they tend to score verbose, professional-sounding answers as "correct" even when numbers have drifted or when minor microdata clues are present.
  - LLMs are **lenient on differencing risks**: they rarely detect when two benign-looking aggregate tables can be subtracted to isolate an individual. Therefore, programmatic state audits must be the source of truth, not LLM judges.

---

## 6. Two-Week vs. Two-Month Roadmap

- **Two More Weeks**:
  - Implement formal $\epsilon$-Differential Privacy via calibrated Laplace noise addition to aggregate means, enabling safe reporting of cohorts with $k < 10$ without complete suppression.
  - Implement fuzzy semantic entity resolution for department names (e.g. mapping "PW" to "Planning & Public Works").
- **Two More Months**:
  - Build a cryptographic secure query execution layer with query differencing history across sessions to block multi-query reconstruction attacks across distributed users.
  - Implement automated differential privacy budget accounting ($\epsilon, \delta$ tracking) stored in a persistent transaction ledger.

---

## 7. What We Deliberately Did Not Do, and Why

1. **Did Not Optimize for Conversational Fluidity**: In high-assurance HR compliance, conversational fluidity encourages helpful models to guess or speculate. Absolute privacy and deterministic refusals must supersede conversational ease.
2. **Did Not Rewrite `src/query.py`**: The prompt strictly constrained touching the underlying aggregation logic. We respected this by building a clean, retry-decorated wrapper around it.
3. **Did Not Rely on Model Self-Censorship for Privacy**: Prompts like "never reveal individuals" fail under adversarial pressure. All privacy guardrails are enforced deterministically in Python before results ever reach the LLM context.
