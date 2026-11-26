"""
Step 14: compare all models. The winner is chosen on validation, test is only reported.
Test labels are made by AI, so every score means agreement with the AI annotator.
Output: data/results/metrics.csv, per_class.csv, by_subreddit.csv, confusion.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import COUNTRIES, LABELS, RESULTS, SAMPLE, SEED, STANCE_COLUMNS
from utils import bootstrap_interval, load_pairs, macro_f1, save_safely

# Simple rule for comparison: against Russia, for Ukraine
RULE = {"russia": "against", "ukraine": "pro"}


def scores(part, labels, rng):
    """Accuracy and macro-F1 with their 95 % intervals."""
    def accuracy(d):
        return accuracy_score(d.label, d.prediction)

    def f1(d):
        return macro_f1(d.label, d.prediction, labels)

    acc_low, acc_high = bootstrap_interval(accuracy, part, rng)
    f1_low, f1_high = bootstrap_interval(f1, part, rng)
    return {"n": len(part), "accuracy": accuracy(part), "acc_low": acc_low, "acc_high": acc_high,
            "macro_f1": f1(part), "f1_low": f1_low, "f1_high": f1_high}


def rows_where_ai_agree():
    """Test comments where both AI annotations agree on both countries."""
    first = pd.read_csv(SAMPLE / "ai_labels_events.csv", keep_default_na=False)
    second = pd.read_csv(SAMPLE / "ai2_labels_test.csv", keep_default_na=False)
    both = second.merge(first, on="row_id", suffixes=("_2", ""))
    same = np.all([both[c] == both[f"{c}_2"] for c in STANCE_COLUMNS], axis=0)
    return set(both.row_id[same])


def load_predictions(gold):
    """All pred_*.csv files together with the AI labels."""
    predictions = pd.concat([
        pd.read_csv(file, keep_default_na=False).assign(model=file.stem.removeprefix("pred_"))
        for file in sorted(RESULTS.glob("pred_*.csv"))])
    return predictions.merge(gold, on=["row_id", "country", "split"], validate="many_to_one")


def evaluate(data, gold, rng):
    """Build the result tables for every model, country and split."""
    agreed = rows_where_ai_agree()
    metrics, per_class, by_subreddit, matrices = [], [], [], []
    for (model, country, split), part in data.groupby(["model", "country", "split"]):
        key = {"model": model, "country": country, "split": split}
        metrics.append({**key, "subset": "all", **scores(part, LABELS, rng)})
        if split != "test":
            continue
        clear = part[part.row_id.isin(agreed)]
        metrics.append({**key, "subset": "both_ai_agree", **scores(clear, LABELS, rng)})
        # Only comments about the country (selected using the AI label)
        about = part[part.label != "na"]
        metrics.append({**key, "subset": "about_country", **scores(about, LABELS[:3], rng)})

        # Recall = what share of a class the model found
        recall = recall_score(part.label, part.prediction, labels=LABELS, average=None, zero_division=0)
        per_class.append({**key, **{f"recall_{label}": r for label, r in zip(LABELS, recall)},
                          **{f"n_{label}": int((part.label == label).sum()) for label in LABELS}})
        for group, rows in part.groupby("sub_group"):
            by_subreddit.append({**key, "sub_group": group, "n": len(rows),
                                 "accuracy": accuracy_score(rows.label, rows.prediction),
                                 "macro_f1": macro_f1(rows.label, rows.prediction)})
        table = pd.DataFrame(confusion_matrix(part.label, part.prediction, labels=LABELS),
                             index=[f"gold {label}" for label in LABELS], columns=LABELS)
        matrices.append(f"== {model} | {country} | test\n{table.to_string()}\n")

    about = gold[(gold.split == "test") & (gold.label != "na")]
    for country, part in about.groupby("country"):
        ruled = part.assign(prediction=RULE[country])
        metrics.append({"model": "country_rule", "country": country, "split": "test",
                        "subset": "about_country", **scores(ruled, LABELS[:3], rng)})
    return pd.DataFrame(metrics), pd.DataFrame(per_class), pd.DataFrame(by_subreddit), matrices


def print_differences(data, winner, others, rng):
    """Paired bootstrap: is the winner better than another model on the same test rows?"""
    print("\nTest macro-F1 difference, winner minus other (95 % interval):")
    for country in COUNTRIES:
        test = data[(data.split == "test") & (data.country == country)]
        wide = test.pivot(index="row_id", columns="model", values="prediction")
        wide["label"] = test.drop_duplicates("row_id").set_index("row_id").label
        for other in others:
            def difference(d):
                return macro_f1(d.label, d[winner]) - macro_f1(d.label, d[other])

            low, high = bootstrap_interval(difference, wide, rng)
            print(f"  {country:8s} vs {other:18s} {difference(wide):+.3f} [{low:+.3f}, {high:+.3f}]")


def main():
    rng = np.random.default_rng(SEED)
    gold = load_pairs()
    data = load_predictions(gold)
    metrics, per_class, by_subreddit, matrices = evaluate(data, gold, rng)

    save_safely(metrics, RESULTS / "metrics.csv")
    save_safely(per_class, RESULTS / "per_class.csv")
    save_safely(by_subreddit, RESULTS / "by_subreddit.csv")
    (RESULTS / "confusion.txt").write_text("\n".join(matrices), encoding="utf-8")

    # The learning-curve models (finetuned_25 ...) do not compete for the winner
    val = metrics[(metrics.split == "val") & ~metrics.model.str.startswith("finetuned_")]
    ranking = val.groupby("model").macro_f1.mean().sort_values(ascending=False)
    print("Validation (mean macro-F1 of both countries):")
    print(ranking.round(3).to_string())
    winner = ranking.index[0]
    print(f"\nWinner on validation: {winner}\n")
    print(metrics[metrics.split == "test"].round(3).to_string(index=False))
    print_differences(data, winner, ranking.index[1:], rng)


if __name__ == "__main__":
    main()
