"""Loads the employee engagement survey dataset.

Source: Pierce County WA employee engagement survey, public open data.
Six waves, 2019-2024. One row per respondent-per-question (long format).

This is the raw export. Nothing has been cleaned. See the README.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "survey_responses.csv"

ANSWER_SCALE = {
    0: "Not Applicable",
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Agree",
    4: "Strongly Agree",
}


def load_survey() -> pd.DataFrame:
    """Return the full survey dataset.

    Raw columns, unmodified from the source export. See README before
    trusting any of them at face value.
    """
    return pd.read_csv(DATA_PATH, low_memory=False)


if __name__ == "__main__":
    df = load_survey()
    print(f"{len(df):,} rows x {len(df.columns)} columns")
    print(df.head())
