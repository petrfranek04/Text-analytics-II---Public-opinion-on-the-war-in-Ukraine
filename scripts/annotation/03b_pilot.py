"""
Step 2: pick 50 comments (10 per event) that a human and the AI annotate
independently, to test the annotation rules (ANOTACE_PRAVIDLA.md).
Output: data/sample/pilot_human.csv (empty label columns, opens in Excel)
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import SAMPLE, SEED
from utils import save_new

PER_EVENT = 10


def main():
    sample = pd.read_csv(SAMPLE / "sample_events_full.csv")
    pilot = sample.groupby("event").sample(n=PER_EVENT, random_state=SEED)
    pilot = pilot.sample(frac=1, random_state=SEED)[["row_id", "text_sentiment"]]
    pilot = pilot.assign(stance_russia="", stance_ukraine="", uncertain="")
    save_new(pilot, SAMPLE / "pilot_human.csv", sep=";", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
