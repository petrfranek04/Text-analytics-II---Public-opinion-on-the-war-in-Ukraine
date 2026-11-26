"""
Step 5: split the sample into train (1300), validation (300) and test (300).
- A whole thread belongs to one part only, so the model cannot "remember" a thread.
- Validation and test have the same event shares as the sample, and subreddit
  shares as close as whole threads allow.
- Threads used in the pilot never go to test.
Labels are not used for splitting.
Output: data/sample/split_events.csv
        data/sample/test_human.csv  (test texts; the second AI annotated them in this order)
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import SAMPLE, SEED
from utils import proportional_quotas, read_excel_csv, save_new

SIZES = {"test": 300, "val": 300}      # the rest is train
GROUPS = ["event", "sub_group"]


def split_threads(sample, pilot_ids):
    result = sample[["row_id", "event", "subreddit", "sub_group", "parent_post_id"]].copy()
    result["split"] = "train"
    pilot_threads = set(result.loc[result["row_id"].isin(pilot_ids), "parent_post_id"])

    def fill(rows, needed):
        """Put shuffled whole threads into the first part where they fit."""
        threads = rows.groupby("parent_post_id").size().sample(frac=1, random_state=SEED)
        for thread, size in threads.items():
            for part in SIZES:
                if part == "test" and thread in pilot_threads:
                    continue
                if size <= needed[part]:
                    result.loc[result["parent_post_id"] == thread, "split"] = part
                    needed[part] -= size
                    break

    # First round: fill the quota of every event x subreddit group
    quotas = {part: proportional_quotas(result.groupby(GROUPS).size(), n) for part, n in SIZES.items()}
    missing = {}
    for group, rows in result.groupby(GROUPS):
        needed = {part: quotas[part][group] for part in SIZES}
        fill(rows, needed)
        event = missing.setdefault(group[0], {part: 0 for part in SIZES})
        for part in SIZES:
            event[part] += needed[part]

    # Second round: what did not fit is taken from other threads of the same event
    for event, needed in missing.items():
        if any(needed.values()):
            fill(result[(result["event"] == event) & (result["split"] == "train")], needed)
            assert not any(needed.values()), f"Could not fill {event}: {needed}"
    return result


def main():
    if (SAMPLE / "split_events.csv").exists():
        print("The split already exists, nothing to do.")
        return
    sample = pd.read_csv(SAMPLE / "sample_events_full.csv")
    pilot = read_excel_csv(SAMPLE / "pilot_human.csv")
    result = split_threads(sample, pilot["row_id"])
    save_new(result, SAMPLE / "split_events.csv")

    test = sample[sample["row_id"].isin(result.loc[result["split"] == "test", "row_id"])]
    test = test.sample(frac=1, random_state=SEED)[["row_id", "text_sentiment"]]
    test = test.assign(stance_russia="", stance_ukraine="", uncertain="")
    save_new(test, SAMPLE / "test_human.csv", sep=";", encoding="utf-8-sig")
    for column in GROUPS:
        print("\n", pd.crosstab(result[column], result["split"], margins=True))


if __name__ == "__main__":
    main()
