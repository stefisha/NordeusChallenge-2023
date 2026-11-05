"""Validate supplied data and submission structure without training models."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "Challenge" / "jobfair_test.csv").open(newline="") as handle:
        expected = [row["club_id"] for row in csv.DictReader(handle)]
    with (ROOT / "league_rank_predictions.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["club_id", "predicted_league_rank"]:
            raise ValueError("Expected club_id and predicted_league_rank columns.")
        rows = list(reader)
    actual = [row["club_id"] for row in rows]
    if len(actual) != len(set(actual)) or actual != expected:
        raise ValueError(
            "Expected exactly one prediction per test club, in test order."
        )
    for row in rows:
        value = float(row["predicted_league_rank"])
        if not value.is_integer() or not 1 <= value <= 14:
            raise ValueError("Predicted positions must be integers from 1 to 14.")
    notebook = json.loads((ROOT / "NordeusChallenge.ipynb").read_text())
    colab = "https://colab.research.google.com/github/stefisha/NordeusChallenge-2023/blob/main/NordeusChallenge.ipynb"
    if colab not in "".join("".join(cell["source"]) for cell in notebook["cells"]):
        raise ValueError("Notebook must link to the current repository.")
    print(
        f"Submission contract verified: {len(rows):,} unique clubs; integer positions 1 to 14."
    )
    print(
        "Current notebook link verified. Models were not trained; test MAE is unknown."
    )


if __name__ == "__main__":
    main()
