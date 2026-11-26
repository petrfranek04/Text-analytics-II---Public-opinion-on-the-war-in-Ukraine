"""
Step 7: two simple models, one per country.
- Majority baseline: always answers the most common label of the training data.
- TF-IDF + logistic regression: learns which words point to which label.
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import COUNTRIES, SEED
from utils import load_pairs, macro_f1, save_predictions

C_VALUES = [0.1, 1, 10]      # regularization strength, chosen on validation


def main():
    pairs = load_pairs()
    train = pairs[pairs.split == "train"]
    evaluation = pairs[pairs.split != "train"].reset_index(drop=True)
    majority = pd.Series(index=evaluation.index, dtype=str)
    tfidf = pd.Series(index=evaluation.index, dtype=str)

    for country in COUNTRIES:
        country_train = train[train.country == country]
        part = evaluation[evaluation.country == country]
        val = part[part.split == "val"]
        majority[part.index] = country_train.label.mode()[0]

        models, scores = {}, {}
        for c in C_VALUES:
            models[c] = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=SEED),
            ).fit(country_train.text, country_train.label)
            scores[c] = macro_f1(val.label, models[c].predict(val.text))
        best = max(scores, key=scores.get)
        print(f"TF-IDF {country}: best C = {best}")
        tfidf[part.index] = models[best].predict(part.text)

    save_predictions("majority", evaluation, majority)
    save_predictions("tfidf", evaluation, tfidf)


if __name__ == "__main__":
    main()
