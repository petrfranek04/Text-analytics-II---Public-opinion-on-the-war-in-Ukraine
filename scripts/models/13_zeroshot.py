"""
Step 9: zero-shot stance (no training). For each comment and country the model
checks four fixed sentences; the one that follows most from the comment wins.
Progress is saved every 20 rows. If stopped, run again to continue.
"""
import sys
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import COUNTRIES, LABELS, RESULTS, ZEROSHOT_MODEL
from utils import load_pairs, prediction_file, save_predictions, save_safely

# Written before seeing any result and not changed afterwards
SENTENCES = {
    "pro": "The author supports {}.",
    "against": "The author criticizes or opposes {}.",
    "neutral": "The author mentions {} without taking a side.",
    "na": "This text is not about {}.",
}
PARTIAL_FILE = RESULTS / "partial_zeroshot.csv"
SAVE_EVERY = 20


def score_rows(rows, done):
    """Add the scores of the four sentences for every row to the list `done`."""
    tokenizer = AutoTokenizer.from_pretrained(ZEROSHOT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(ZEROSHOT_MODEL).eval()
    entailment = model.config.label2id["entailment"]
    for number, row in enumerate(rows, 1):
        sentences = [SENTENCES[label].format(COUNTRIES[row.country]) for label in LABELS]
        inputs = tokenizer([row.text] * len(LABELS), sentences, padding=True,
                           truncation="only_first", max_length=512, return_tensors="pt")
        with torch.no_grad():
            scores = model(**inputs).logits[:, entailment].softmax(dim=0).tolist()
        done.append({"row_id": row.row_id, "country": row.country, **dict(zip(LABELS, scores))})
        if number % SAVE_EVERY == 0 or number == len(rows):
            save_safely(pd.DataFrame(done), PARTIAL_FILE)
            print(f"  {number}/{len(rows)} done", flush=True)


def main():
    if prediction_file("zeroshot").exists():
        print("zero-shot already done, skipping")
        return
    pairs = load_pairs()
    pairs = pairs[pairs.split != "train"].reset_index(drop=True)

    done = pd.read_csv(PARTIAL_FILE).to_dict("records") if PARTIAL_FILE.exists() else []
    finished = {(row["row_id"], row["country"]) for row in done}
    todo = [row for row in pairs.itertuples() if (row.row_id, row.country) not in finished]
    print(f"zero-shot: {len(finished)} done, {len(todo)} to go", flush=True)
    if todo:
        score_rows(todo, done)

    scores = pd.DataFrame(done).set_index(["row_id", "country"])
    best = scores.loc[list(zip(pairs.row_id, pairs.country)), LABELS].idxmax(axis=1)
    save_predictions("zeroshot", pairs, best.values)


if __name__ == "__main__":
    main()
