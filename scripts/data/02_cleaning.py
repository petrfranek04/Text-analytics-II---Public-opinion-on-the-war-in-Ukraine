"""
Preparation step 2: clean the raw data.
Output: data/reddit_cleaned_full.csv
        data/sample/sample_anonymized.csv (100 rows without the raw text)
If the cleaned file exists, nothing happens: a row's number is its ID in the labels.
Known limits: "I'm a bot" and "[removed]" inside a longer text are not caught.
"""
import sys
from pathlib import Path
import re

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import CLEAN_FILE, RAW_FILE, SAMPLE, SEED
from utils import save_new

BOT_PHRASES = r"I am a bot|performed automatically|RemindMe!|submission statement"


def clean_for_sentiment(text):
    """Keeps punctuation and capital letters, because sentiment models use them."""
    # Lines starting with ">" quote someone else, often an opinion the author rejects
    text = re.sub(r"(?m)^>.*$", "", str(text))
    text = re.sub(r"http\S+|www\.\S+", "", text)           # web links
    return re.sub(r"\s+", " ", text).strip()               # extra spaces


def clean_for_topics(text):
    """Also removes subreddit and user names and formatting characters."""
    text = re.sub(r"/?[ru]/[A-Za-z0-9_-]+", "", text)
    text = re.sub(r"[\*\[\]\(\)~_`#]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean(data):
    """Returns the cleaned data and the number of rows after every step."""
    counts = {"raw records": len(data)}

    data = data.drop_duplicates(subset=["record_id"])
    counts["without duplicate IDs"] = len(data)

    data = data[data["author_hash"] != "deleted_or_bot"]
    data = data[~data["text"].isin(["[removed]", "[deleted]", ""])]
    data = data[~data["text"].str.contains(BOT_PHRASES, case=False, na=False)].copy()
    counts["without deleted and bot messages"] = len(data)

    data["text_sentiment"] = data["text"].apply(clean_for_sentiment)
    data["text_topic"] = data["text_sentiment"].apply(clean_for_topics)

    data = data.drop_duplicates(subset=["text_sentiment"]).copy()      # copied spam
    counts["without repeated texts"] = len(data)

    data["word_count"] = data["text_sentiment"].apply(lambda text: len(text.split()))
    data = data[data["word_count"] >= 5]                               # e.g. "Agreed!"
    counts["with at least 5 words"] = len(data)
    return data, counts


def main():
    if CLEAN_FILE.exists():
        print("The cleaned data already exists, nothing to do.")
        return
    # keep_default_na=False: an empty text stays "" and is not turned into a missing value
    raw = pd.read_csv(RAW_FILE, keep_default_na=False,
                      dtype={"parent_post_id": str, "author_hash": str, "text": str})
    cleaned, counts = clean(raw)
    for step, number in counts.items():
        print(f"{step:35s} {number}")
    save_new(cleaned, CLEAN_FILE)
    example = cleaned.drop(columns=["text"]).sample(n=min(100, len(cleaned)), random_state=SEED)
    save_new(example, SAMPLE / "sample_anonymized.csv")


if __name__ == "__main__":
    main()
