# SurveyIQ — Stateful, Privacy-Preserving HR Survey AI Agent

A production-grade, stateful AI agent built with **LangGraph** and **Claude 3.5 Sonnet** (`claude-3-5-sonnet-20241022`) to query the Pierce County WA Employee Engagement Survey dataset (132,549 rows across 6 waves).

The system enforces strict differential privacy guarantees ($k$-anonymity, differencing attack defense, dynamic fallback), native fault tolerance via **tenacity** retries, a dedicated pre-LLM refusal router boundary, deterministic loop safety caps, and multi-hop comparative reasoning.

---

## Architecture Overview

```
[START]
   │
   ▼
[router_node] ─── (Individual targeting / raw data / out-of-scope) ───► [END] (Immediate Refusal)
   │
   ▼ (Valid aggregate query)
[agent_node] (Claude 3.5 Sonnet) ◄────────────────────────────────────────+
   │                                                                      │
   ▼ (route_after_llm)                                                    │
   ├────── (step_count >= 4) ───────────► [terminal_budget_node] ──► [END]│
   ├────── (has tool_calls) ────────────► [tool_node] ────────────────────+
   │                                            │
   │                                            │ (Tenacity retry +
   │                                            │  k-Anonymity guardrail +
   │                                            │  Dynamic Fallback +
   │                                            │  step_count += 1)
   ▼ (no tool_calls / final answer)
 [END]
```

---

## Architectural Highlights

1. **Dedicated Refusal Router Boundary (`router_node`)**:
   Analyzes incoming user queries before any LLM generation or database execution. Instantly returns a hard refusal for:
   - Queries targeting individuals (e.g. "Who is the employee who gave score 1?", "What did Bob say?").
   - Raw microdata/record extraction (e.g. "Dump all rows", "Export unaggregated CSV").
   - Out-of-scope requests (e.g. salaries, county budget, external topics).

2. **Fault Tolerance via Tenacity (`@retry`)**:
   Wraps the unedited `src.query.query_aggregate` with `tenacity.retry(wait_exponential, stop_after_attempt(5))` to natively absorb the simulated 12% transient dropouts.

3. **$k$-Anonymity Privacy Guardrail ($k=10$) & Dynamic Fallback**:
   Intercepts `respondent_count`. If $1 \le n \le 9$, exact scores and distributions are suppressed to prevent differencing attacks. Instead of failing, it logs a `privacy_block` in `metadata_log` and executes a **Dynamic Fallback**: automatically stripping the most specific filter (`role`) and re-querying the broader department (e.g. Assessor-Treasurer's Office Manager $n=1 \to$ Department $n=45$).

4. **Long-Format 17-Question Detection**:
   In this dataset, respondents answer up to 17 questions in long format. When `question=None`, a single respondent yields 17 rows. Our tool audits effective respondents (`rows // 17`), preventing single individuals from being mistakenly exposed under aggregate queries.

5. **Role Normalization Across Waves**:
   Transparently maps `"Staff"` (2019–2020) and `"Staff Member"` (2021–2024) to eliminate spurious empty results.

6. **Loop Safety Operational Budget**:
   Conditional edge caps execution at `step_count >= 4`. At the boundary, it terminates immediately with:
   `"Terminal error: Operational budget is exhausted (step count >= 4). Execution halted to prevent runaway loops."`

7. **Multi-Hop Comparative Reasoning**:
   The agent prompt enforces sequential tool invocations across distinct entities (e.g. comparing Human Services vs District Court) before synthesizing a quantitative comparative response.

---

## Setup & Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt langchain-anthropic

# 2. (Optional) Configure Anthropic API Key for live Claude 3.5 Sonnet execution
# If unset, the system seamlessly uses the local deterministic test agent.
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

---

## Running the Evaluation & Tests

### Run the Evaluation Battery (Deliverables 2 & 3)
```bash
python -m src.evaluation
```
Executes the comprehensive test battery across routing, privacy guardrails, multi-hop reasoning, and loop safety, followed by the written evaluation self-critiques and red-team attack reports.

### Run the Unit & Integration Test Suite
```bash
python -m pytest -v tests/
```
Runs all 32 automated tests covering:
- `tests/test_router.py`: Refusal boundaries and approval classification.
- `tests/test_tools.py`: Tenacity retry recovery, $k$-anonymity suppression, dynamic fallback, and role normalization.
- `tests/test_loop_safety.py`: Deterministic halting at `step_count >= 4`.
- `tests/test_graph.py`: End-to-end multi-hop graph execution.
- `tests/test_red_team.py`: The 3 adversarial red-team attack vectors.

### Create Submission Bundle
```bash
python bundle.py
# Or via Makefile:
make bundle
```
Creates `sphesihle-anthony-mhlongo-takehome.zip` containing all source code, tests, dataset, documentation, and `TRADEOFFS.md`.

---

## Project Structure

```
├── TRADEOFFS.md             # 2-page writeup on architecture, dataset findings & tradeoffs
├── README.md                # System documentation and execution guide
├── requirements.txt         # Project dependencies
├── Makefile                 # Make targets for install, test, eval, bundle
├── bundle.py                # Zip packaging script
├── data/
│   ├── loader.py            # Dataset loader
│   └── survey_responses.csv # Pierce County survey dataset (132,549 rows)
├── src/
│   ├── __init__.py          # Package initialization
│   ├── state.py             # AgentState (TypedDict) and metadata log schemas
│   ├── query.py             # Raw query tool (provided, unedited)
│   ├── tools.py             # Tenacity retry wrapper, k-anonymity, dynamic fallback
│   ├── router.py            # Dedicated refusal router node
│   ├── agent.py             # LangGraph workflow, Claude 3.5 Sonnet binding, loop budget
│   ├── mock_llm.py          # Deterministic local model for offline testing
│   └── evaluation.py        # Comprehensive evaluation battery & red-team reports
└── tests/
    ├── test_router.py       # Unit tests for router refusal boundaries
    ├── test_tools.py        # Unit tests for tenacity retries and k-anonymity
    ├── test_loop_safety.py  # Unit tests for loop safety budget cap
    ├── test_graph.py        # Integration tests for graph execution
    └── test_red_team.py     # Adversarial attack tests
```
