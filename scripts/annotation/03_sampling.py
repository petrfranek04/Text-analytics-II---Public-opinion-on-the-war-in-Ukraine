"""
Step 1: draw the random sample of 1900 comments for annotation.
- Candidates: comments with 5-150 words that mention Ukraine or Russia.
- Each event x subreddit group gets the same share in the sample as among the candidates.
- At most 20 comments from one thread.
Output: data/sample/sample_events_full.csv      (with metadata)
        data/sample/sample_events_to_label.csv  (texts only, for blind annotation)
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import CLEAN_FILE, RUS_PATTERN, SAMPLE, SEED, UKR_PATTERN
from utils import proportional_quotas, save_new

SAMPLE_SIZE = 1900
MAX_PER_THREAD = 20
GROUPS = ["event", "sub_group"]


def load_candidates():
    data = pd.read_csv(CLEAN_FILE)
    data["row_id"] = data.index      # stable ID: the row number in the cleaned file
    data = data.dropna(subset=["text_sentiment"])
    data = data[(data["type"] == "comment") & data["word_count"].between(5, 150)]
    mentions = data["text_sentiment"].str.contains(f"{UKR_PATTERN}|{RUS_PATTERN}", case=False)
    data = data[mentions].copy()
    # The two US political subreddits are small, so they form one group
    data["sub_group"] = data["subreddit"].replace({"politics": "US", "Conservative": "US"})
    return data


def draw_sample(candidates):
    quotas = proportional_quotas(candidates.groupby(GROUPS).size(), SAMPLE_SIZE)
    # Shuffle, then keep only the first 20 comments of every thread
    shuffled = candidates.sample(frac=1, random_state=SEED)
    capped = shuffled.groupby(GROUPS + ["parent_post_id"]).head(MAX_PER_THREAD)
    parts = []
    for group, rows in capped.groupby(GROUPS):
        parts.append(rows.sample(n=min(quotas[group], len(rows)), random_state=SEED))
    # Shuffle again, so the annotator does not see one event after another
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def main():
    if (SAMPLE / "sample_events_full.csv").exists():
        print("The sample already exists, nothing to do.")
        return
    sample = draw_sample(load_candidates())
    columns = ["row_id", "event", "subreddit", "sub_group", "parent_post_id", "text_sentiment"]
    save_new(sample[columns], SAMPLE / "sample_events_full.csv")
    to_label = sample[["row_id", "text_sentiment"]].assign(stance_russia="", stance_ukraine="", uncertain="")
    save_new(to_label, SAMPLE / "sample_events_to_label.csv")
    print(pd.crosstab(sample["event"], sample["sub_group"], margins=True))


if __name__ == "__main__":
    main()
