"""
Step 4: turn the AI annotation codes into the label table.
The AI (Claude) annotated all 1900 comments on 25 Nov 2026 following
ANOTACE_PRAVIDLA.md, seeing only the text (no event or subreddit).
These are AI labels, not verified truth.
Output: data/sample/ai_labels_events.csv
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import SAMPLE
from utils import decode_labels, save_new


def main():
    texts = pd.read_csv(SAMPLE / "sample_events_to_label.csv", keep_default_na=False)
    labels = decode_labels(SAMPLE / "ai_labels_progress.txt", len(texts))
    result = pd.concat([texts[["row_id", "text_sentiment"]], labels], axis=1)
    save_new(result, SAMPLE / "ai_labels_events.csv")
    for column in labels.columns:
        print("\n" + labels[column].value_counts().to_string())


if __name__ == "__main__":
    main()
