"""
Step 3: agreement between the human and the AI on the 50 pilot comments.
Cohen's kappa = agreement corrected for chance (1 = perfect, 0 = random).
Output: data/sample/pilot_disagreements.csv
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import LABELS, SAMPLE, STANCE_COLUMNS
from utils import read_excel_csv, save_safely


def main():
    if not (SAMPLE / "pilot_ai.csv").exists():
        print("The pilot files are not available (they contain comment texts), skipping.")
        return
    human = read_excel_csv(SAMPLE / "pilot_human.csv")
    ai = read_excel_csv(SAMPLE / "pilot_ai.csv")
    assert human[STANCE_COLUMNS].isin(LABELS).all().all(), "Invalid or empty label in pilot_human.csv"
    data = human.merge(ai, on="row_id", suffixes=("_human", "_ai"), validate="one_to_one")
    assert len(data) == len(human) == len(ai), "The two files contain different rows"

    for column in STANCE_COLUMNS:
        h, a = data[f"{column}_human"], data[f"{column}_ai"]
        print(f"\n{column}: agreement {(h == a).mean():.2f}, kappa {cohen_kappa_score(h, a):.2f}")
        print(pd.DataFrame(confusion_matrix(h, a, labels=LABELS),
                           index=[f"human {label}" for label in LABELS], columns=LABELS))

    different = ((data.stance_russia_human != data.stance_russia_ai)
                 | (data.stance_ukraine_human != data.stance_ukraine_ai))
    columns = ["row_id", "text_sentiment_ai", "stance_russia_human", "stance_russia_ai",
               "stance_ukraine_human", "stance_ukraine_ai", "uncertain_human", "uncertain_ai"]
    result = data.loc[different, columns].rename(columns={"text_sentiment_ai": "text"})
    print(f"\nComments with a disagreement: {len(result)} of {len(data)}")
    save_safely(result, SAMPLE / "pilot_disagreements.csv")


if __name__ == "__main__":
    main()
