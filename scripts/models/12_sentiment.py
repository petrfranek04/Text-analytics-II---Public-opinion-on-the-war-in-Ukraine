"""
Step 8: tone used as stance, with VADER (word list) and a RoBERTa sentiment model.
positive -> pro, negative -> against, neutral -> neutral.
A country whose keywords do not appear in the comment gets "na".
"""
import sys
from pathlib import Path
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import RUS_PATTERN, SENTIMENT_MODEL, UKR_PATTERN
from utils import load_pairs, prediction_file, save_predictions

PATTERNS = {"russia": RUS_PATTERN, "ukraine": UKR_PATTERN}
TONE_TO_STANCE = {"positive": "pro", "negative": "against", "neutral": "neutral"}
BATCH = 16


def vader_tones(texts):
    vader = SentimentIntensityAnalyzer()
    tones = {}
    for text in texts:
        score = vader.polarity_scores(text)["compound"]      # from -1 to +1
        if score >= 0.05:
            tones[text] = "positive"
        elif score <= -0.05:
            tones[text] = "negative"
        else:
            tones[text] = "neutral"
    return tones


def roberta_tones(texts):
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).eval()
    tones = {}
    for start in range(0, len(texts), BATCH):
        batch = texts[start:start + BATCH]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            best = model(**inputs).logits.argmax(dim=1)
        for text, index in zip(batch, best.tolist()):
            tones[text] = model.config.id2label[index].lower()
    return tones


def to_stance(pairs, tones):
    stances = []
    for row in pairs.itertuples():
        mentioned = re.search(PATTERNS[row.country], row.text, re.IGNORECASE)
        stances.append(TONE_TO_STANCE[tones[row.text]] if mentioned else "na")
    return stances


def main():
    pairs = load_pairs()
    pairs = pairs[pairs.split != "train"].reset_index(drop=True)
    texts = list(pairs.text.unique())
    for name, get_tones in [("vader", vader_tones), ("sentiment_roberta", roberta_tones)]:
        if prediction_file(name).exists():
            print(name, "already done, skipping")
        else:
            save_predictions(name, pairs, to_stance(pairs, get_tones(texts)))


if __name__ == "__main__":
    main()
