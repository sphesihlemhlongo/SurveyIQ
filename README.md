# Starter repo — Senior AI Developer take-home

Read the assignment document first. This repo gives you the dataset access
layer so your time goes into architecture, evaluation, and investigation —
not plumbing.

## Setup

```bash
pip install -r requirements.txt
python data/loader.py   # sanity check the dataset loads
```

You will need an API key for whichever model provider you use. Put it in
`.env`; do not commit it.

## What's here

```
data/survey_responses.csv   the dataset (132,549 rows)
data/loader.py              loads it — done, don't rewrite
src/query.py                the only tool available for touching the
                             dataset. Fully implemented — use it as-is,
                             don't reimplement the aggregation.
```

Not here, and yours to create: the orchestration/agent code, the
evaluation, `TRADEOFFS.md`, and your own README section explaining how to
run what you built.

## The dataset

Pierce County WA employee engagement survey, public open data. Six waves
(2019–2024), 17 engagement statements, 23 departments, 5 roles.

Long format: **one row per respondent per question.** There is no
respondent ID, so rows cannot be grouped back into individuals directly.

This is the raw export, unmodified. We have not cleaned anything.

| Column | Notes |
|---|---|
| `Year` | survey wave, e.g. `2024 May` |
| `Status` | `Complete` or `Partial` |
| `Role` | see below |
| `Department` | 23 values; nullable |
| `Director`, `Manager`, `Supervisor`, `Lead`, `Staff` | boolean flags |
| `Question` | one of 17 numbered statements |
| `Answer_Numeric` | 0–4 |
| `Answer_Text` | text form of the above; nullable |

Answer scale: `0` Not Applicable, `1` Strongly Disagree, `2` Disagree,
`3` Agree, `4` Strongly Agree. This is a 4-point agreement scale with a
separate N/A option, not a 5-point Likert with a neutral midpoint.

We are not going to tell you anything else about this data's shape,
including whether the columns above are reliable as given. Look at it
before you build against it.

## About `query_aggregate`

It's fully implemented in `src/query.py`. Read it before you build around
it — in particular, read what it can raise, and think about when.
