"""Functions shared by several scripts: files, labels, scores."""
import os
import re

import numpy as np
import pandas as pd
import truststore
from sklearn.metrics import accuracy_score, f1_score

from config import COUNTRIES, LABELS, N_BOOT, RESULTS, SAMPLE, STANCE_COLUMNS

# Check HTTPS certificates against the operating system's store (needed for model
# downloads behind antivirus HTTPS scanning). Verification stays on.
truststore.inject_into_ssl()


# ---------- Files ----------
def save_new(df, path, **options):
    """Save a CSV only if it does not exist yet. Labels depend on these files."""
    if path.exists():
        print(f"Not overwritten, the file already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **options)
    print(f"Saved {path} ({len(df)} rows)")


def save_safely(df, path):
    """Overwrite a CSV through a temporary file, so stopping never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_excel_csv(path):
    """Files filled in by hand in Excel use ';' and UTF-8 with BOM."""
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", keep_default_na=False)


# ---------- Labels ----------
def decode_labels(path, n_rows):
    """Turn short AI annotation codes into columns.
    One code = row number + Russia + Ukraine (p/a/n/x) + uncertain (0/1), e.g. 17ax0."""
    letters = {"p": "pro", "a": "against", "n": "neutral", "x": "na"}
    tokens = path.read_text(encoding="utf-8").split()
    parsed = [re.fullmatch(r"(\d+)([panx])([panx])([01])", token) for token in tokens]
    assert all(parsed), f"Invalid code in {path.name}"
    assert [int(m[1]) for m in parsed] == list(range(n_rows)), "Row numbers do not match the texts"
    return pd.DataFrame({"stance_russia": [letters[m[2]] for m in parsed],
                         "stance_ukraine": [letters[m[3]] for m in parsed],
                         "uncertain": [int(m[4]) for m in parsed]})


def load_pairs():
    """One row per comment and country: row_id, text, split, event, ..., country, label."""
    split = pd.read_csv(SAMPLE / "split_events.csv")
    labels = pd.read_csv(SAMPLE / "ai_labels_events.csv", keep_default_na=False)
    data = split.merge(labels, on="row_id", validate="one_to_one")
    data = data.rename(columns={"text_sentiment": "text"})
    pairs = data.melt(id_vars=["row_id", "text", "split", "event", "sub_group", "parent_post_id"],
                      value_vars=STANCE_COLUMNS, var_name="country", value_name="label")
    pairs["country"] = pairs["country"].str.replace("stance_", "")
    return pairs


# ---------- Sampling ----------
def proportional_quotas(sizes, total):
    """Split `total` in proportion to group sizes (largest remainder method)."""
    exact = sizes * total / sizes.sum()
    quotas = exact.astype(int)
    remainders = (exact - quotas).sort_values(ascending=False, kind="stable")
    quotas[remainders.index[:total - quotas.sum()]] += 1
    return quotas


# ---------- Scores ----------
def macro_f1(gold, predicted, labels=LABELS):
    """Average F1 over the classes, so rare classes count as much as common ones."""
    return f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)


def bootstrap_interval(score, data, rng):
    """95 % interval of score(data): recompute it on random samples of the rows."""
    values = [score(data.iloc[rng.integers(0, len(data), len(data))]) for _ in range(N_BOOT)]
    return np.percentile(values, [2.5, 97.5])


def prediction_file(model):
    return RESULTS / f"pred_{model}.csv"


def save_predictions(model, pairs, predictions):
    """Save predictions for validation and test, print the validation score only."""
    data = pairs.assign(prediction=list(predictions))
    save_safely(data[["row_id", "country", "split", "prediction"]], prediction_file(model))
    for country in COUNTRIES:
        val = data[(data.split == "val") & (data.country == country)]
        print(f"{model} | validation | {country}: accuracy {accuracy_score(val.label, val.prediction):.3f}, "
              f"macro-F1 {macro_f1(val.label, val.prediction):.3f}", flush=True)
