"""
Step 6: agreement between two independent AI annotations of the test set.
The second one was made by a separate AI agent that saw only the rules, the pilot
examples and the texts. Agreement shows consistency, NOT correctness.
Output: data/sample/ai2_labels_test.csv, ai_ai_disagreements.csv
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import LABELS, SAMPLE, STANCE_COLUMNS
from utils import decode_labels, read_excel_csv, save_new, save_safely


def main():
    test = read_excel_csv(SAMPLE / "test_human.csv")[["row_id", "text_sentiment"]]
    second = pd.concat([test, decode_labels(SAMPLE / "ai2_codes_test.txt", len(test))], axis=1)
    save_new(second, SAMPLE / "ai2_labels_test.csv")

    first = pd.read_csv(SAMPLE / "ai_labels_events.csv", keep_default_na=False)
    data = first.merge(second.drop(columns="text_sentiment"), on="row_id", suffixes=("_1", "_2"),
                       validate="one_to_one")
    for column in STANCE_COLUMNS:
        one, two = data[f"{column}_1"], data[f"{column}_2"]
        print(f"\n{column}: agreement {(one == two).mean():.2f}, kappa {cohen_kappa_score(one, two):.2f}")
        print(pd.DataFrame(confusion_matrix(one, two, labels=LABELS),
                           index=[f"AI1 {label}" for label in LABELS], columns=LABELS))

    same = ((data.stance_russia_1 == data.stance_russia_2)
            & (data.stance_ukraine_1 == data.stance_ukraine_2))
    print(f"\nAgreement on both countries: {same.mean():.2f}")
    save_safely(data[~same], SAMPLE / "ai_ai_disagreements.csv")


if __name__ == "__main__":
    main()
