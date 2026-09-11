# SurveyIQ — Privacy-Preserving HR Survey AI Agent

A stateful AI agent built with **LangGraph** and **Claude Sonnet 4.5** (`claude-sonnet-4-5-20250929`) to query the Pierce County WA Employee Engagement Survey dataset (132,549 rows across 6 waves).

Enforces $k$-anonymity, blocks differencing attacks, executes dynamic fallback, handles transient query errors via Tenacity retries, blocks unauthorized queries with a fast pre-LLM router, caps loops, and answers multi-hop comparative questions.

---

## Architecture Overview

```
[START]
   │
   ▼
[router_node] ─── (Individual targeting / raw data / out-of-scope) ───► [END] (Immediate Refusal)
   │
   ▼ (Valid aggregate query)
[agent_node] (Claude Sonnet 4.5) ◄────────────────────────────────────────+
   │                                                                      │
   ▼ (route_after_llm)                                                    │
   ├────── (step_count >= 4) ───────────► [terminal_budget_node] ──► [END]│
   ├────── (has tool_calls) ────────────► [tool_node] ────────────────────+
   │                                            │
   │                                            │ (Tenacity retry +
   │                                            │  k-Anonymity guardrail +
   │                                            │  Dynamic Fallback +
   │                                            │  step_count += n)
   ▼ (no tool_calls / final answer)
 [END]
```

---

## Architectural Pillars

1. **Pre-LLM Refusal Router (`router_node`)**:
   Classifies incoming queries before running model calls or database queries. Instantly halts and refuses:
   - Individual targeting ("Who gave score 1 on question 4?", "What did Bob say?").
   - Raw record extraction ("Dump all rows", "Export CSV").
   - Out-of-scope questions (salaries, county budget, outside knowledge).

2. **Fault Tolerance via Tenacity (`@retry`)**:
   Wraps `src.query.query_aggregate` with `tenacity.retry(wait_exponential, stop_after_attempt(5))` to absorb the simulated 12% transient dropouts.

3. **$k$-Anonymity Guardrail ($k=10$) & Dynamic Fallback**:
   Checks `respondent_count`. If $1 \le n \le 9$, suppresses exact scores to prevent differencing attacks. Logs `privacy_block` in `metadata_log`. Executes **Dynamic Fallback**: drops the `role` filter and re-queries the broader department level (e.g., Assessor-Treasurer Manager $n=1 \to$ Department $n=45$). Discloses the generalization to the user.

4. **17-Question Long-Format Detection**:
   The dataset stores one row per question per respondent (17 questions per survey wave). When `question=None`, 1 respondent yields 17 rows. The tool calculates effective respondents (`rows // 17`), preventing single-person cohorts from leaking under unconstrained aggregate queries.

5. **Role Normalization Across Waves**:
   Maps `"Staff"` (2019–2020) and `"Staff Member"` (2021–2024) to eliminate spurious empty results.

6. **Loop Safety Operational Budget**:
   Caps execution at `step_count >= 4`. At threshold, halts and outputs:
   `"Terminal error: Operational budget is exhausted (step count >= 4). Execution halted to prevent runaway loops."`

7. **Multi-Hop Comparative Reasoning**:
   Executes distinct sequential query calls across entities (e.g., Human Services vs District Court) before generating a quantitative comparative synthesis.

---

## Setup & Installation

### 1. Create & Activate Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (Git Bash):**
```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt langchain-anthropic
```

### 3. (Optional) Configure Anthropic API Key

If unset, the system uses the local deterministic test agent (`mock_llm.py`).

**Via `.env` file (recommended):**
Copy `.env.example` to `.env` and insert your key:
```bash
cp .env.example .env
```
*(On Windows PowerShell: `Copy-Item .env.example .env`)*

**Or export directly in shell:**

- **macOS / Linux / Git Bash:**
  ```bash
  export ANTHROPIC_API_KEY="your-anthropic-api-key"
  ```
- **Windows (PowerShell):**
  ```powershell
  $env:ANTHROPIC_API_KEY="your-anthropic-api-key"
  ```
- **Windows (Command Prompt):**
  ```cmd
  set ANTHROPIC_API_KEY="your-anthropic-api-key"
  ```

---

## Running the Evaluation & Tests

### Run the Evaluation Battery (Deliverables 2 & 3)
```bash
python -m src.evaluation
```
Runs test cases across routing, privacy guardrails, multi-hop reasoning, and loop safety, followed by the written self-critiques and red-team attack reports.

### Run the Unit & Integration Test Suite
```bash
python -m pytest -v tests/
```
Runs 32 tests covering:
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
Creates `sphesihle-anthony-mhlongo-takehome.zip` containing project code, tests, dataset, documentation, and `TRADEOFFS.md`.

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
│   ├── agent.py             # LangGraph workflow, Claude Sonnet 4.5 binding, loop budget
│   ├── mock_llm.py          # Deterministic local model for offline testing
│   └── evaluation.py        # Evaluation battery & red-team reports
└── tests/
    ├── test_router.py       # Unit tests for router refusal boundaries
    ├── test_tools.py        # Unit tests for tenacity retries and k-anonymity
    ├── test_loop_safety.py  # Unit tests for loop safety budget cap
    ├── test_graph.py        # Integration tests for graph execution
    └── test_red_team.py     # Adversarial attack tests
```
