# ============================================================
# BLOK 6: METRIKY, DETEKCE CÍLE A TABULKA CHYB
# Co dělá:
#   1) spojí oba gold vzorky (náhodný 400 + cílený 150) a doplní
#      předpovědi VADER/RoBERTa i pro cílený vzorek
#   2) SENTIMENT VŮČI CÍLI: u komentářů, kde je gold cíl Ukrajina nebo
#      Rusko, porovná gold sentiment s předpovědí modelu (accuracy,
#      macro-F1 + 95% bootstrap interval), zvlášť pro každý typ vzorku
#   3) DETEKCE CÍLE: baseline podle klíčových slov, vyhodnocený na
#      náhodném vzorku (4 třídy: ukraine / russia / both / other)
#   4) uloží tabulku chyb RoBERTy pro ruční error analysis
# Vstup:  gold_with_predictions.csv, sample_full.csv,
#         gold_labels_targeted.csv, sample_targeted_full.csv
# Výstup: all_gold_predictions.csv, errors_roberta.csv
# Požadavky: spustit ve stejné session po Bloku 5 (používá objekty
#            `vader` a `clf`, které blok 5 vytvořil).
# ============================================================
import pandas as pd
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE = "/content/drive/MyDrive/ukraine_project"
rng = np.random.default_rng(42)
SENT = ["negative", "neutral", "positive"]

# ---------- 1) Načtení a spojení vzorků ----------
g1 = pd.read_csv(f"{BASE}/data/sample/gold_with_predictions.csv")
meta1 = pd.read_csv(f"{BASE}/data/sample/sample_full.csv")[["row_id", "event", "subreddit", "sample_type"]]
g1 = g1.merge(meta1, on="row_id", how="left")

g2 = pd.read_csv(f"{BASE}/data/sample/gold_labels_targeted.csv").drop(columns=["sample_type"])
meta2 = pd.read_csv(f"{BASE}/data/sample/sample_targeted_full.csv")[["row_id", "event", "subreddit"]]
g2 = g2.merge(meta2, on="row_id", how="left")
g2["sample_type"] = "targeted_ua"

def add_predictions(df):
    texts = df["text_sentiment"].tolist()
    df["vader_compound"] = [vader.polarity_scores(t)["compound"] for t in texts]
    df["vader_label"] = df["vader_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral"))
    res = clf(texts, batch_size=32, truncation=True, max_length=512)
    for k in SENT:
        df[f"rob_{k}"] = [{d["label"].lower(): d["score"] for d in r}[k] for r in res]
    cols = [f"rob_{k}" for k in SENT]
    df["roberta_label"] = df[cols].idxmax(axis=1).str.replace("rob_", "")
    df["roberta_score"] = df["rob_positive"] - df["rob_negative"]
    return df

g2 = add_predictions(g2)
all_gold = pd.concat([g1, g2], ignore_index=True)
all_gold.to_csv(f"{BASE}/data/sample/all_gold_predictions.csv", index=False)
print("Celkem komentářů:", len(all_gold))
print(all_gold["sample_type"].value_counts().to_dict())

# ---------- pomocné funkce ----------
def boot_ci(y_true, y_pred, metric, n=1000):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(metric(y_true[idx], y_pred[idx]))
    return np.percentile(vals, [2.5, 97.5])

def evaluate(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    f1 = lambda a, b: f1_score(a, b, labels=labels, average="macro", zero_division=0)
    mf1 = f1(y_true, y_pred)
    a_lo, a_hi = boot_ci(y_true, y_pred, accuracy_score)
    f_lo, f_hi = boot_ci(y_true, y_pred, f1)
    return acc, a_lo, a_hi, mf1, f_lo, f_hi

# ---------- 2) Sentiment vůči cíli (jen gold cíl = ukraine/russia) ----------
sub = all_gold[all_gold["target"].isin(["ukraine", "russia"])].copy()
print("\n=== SENTIMENT VŮČI CÍLI (gold cíl = Ukrajina nebo Rusko), n =", len(sub), "===")

groups = {
    "random (náhodný)": sub[sub.sample_type == "random"],
    "sarcasm_hint": sub[sub.sample_type == "sarcasm_hint"],
    "targeted_ua": sub[sub.sample_type == "targeted_ua"],
    "VŠE spojené": sub,
}
rows = []
for gname, gdf in groups.items():
    if len(gdf) < 10:
        continue
    for model in ["vader_label", "roberta_label"]:
        acc, a_lo, a_hi, mf1, f_lo, f_hi = evaluate(gdf["target_sentiment"], gdf[model], SENT)
        rows.append({"vzorek": gname, "n": len(gdf), "model": model.replace("_label", ""),
                     "accuracy": round(acc, 3), "acc 95% CI": f"{a_lo:.2f}-{a_hi:.2f}",
                     "macro-F1": round(mf1, 3), "F1 95% CI": f"{f_lo:.2f}-{f_hi:.2f}"})
print(pd.DataFrame(rows).to_string(index=False))

print("\nRoBERTa - confusion matrix (řádky = gold, sloupce = předpověď):")
print(pd.DataFrame(confusion_matrix(sub["target_sentiment"], sub["roberta_label"], labels=SENT),
                   index=[f"gold_{s}" for s in SENT], columns=[f"pred_{s}" for s in SENT]))
print("\nVADER - confusion matrix:")
print(pd.DataFrame(confusion_matrix(sub["target_sentiment"], sub["vader_label"], labels=SENT),
                   index=[f"gold_{s}" for s in SENT], columns=[f"pred_{s}" for s in SENT]))
print("\nMajoritní třída (triviální baseline) accuracy:",
      round(sub["target_sentiment"].value_counts(normalize=True).iloc[0], 3))

# ---------- 3) Detekce cíle: baseline podle klíčových slov ----------
UKR = r"ukrain|kyiv|kiev|zelensk|kharkiv|kherson|donbas|mariupol|bakhmut|azov|avdiivka|kramatorsk"
RUS = r"russia|putin|moscow|kremlin|ruzzia|vatnik|wagner|kursk|rosgvard|\bru\b"

def keyword_target(t):
    u = bool(re.search(UKR, t, flags=re.I))
    r = bool(re.search(RUS, t, flags=re.I))
    if u and r: return "both"
    if u: return "ukraine"
    if r: return "russia"
    return "other"

all_gold["pred_target_kw"] = all_gold["text_sentiment"].apply(keyword_target)
TGT = ["ukraine", "russia", "both", "other"]
rnd = all_gold[all_gold.sample_type == "random"]
print("\n=== DETEKCE CÍLE (klíčová slova), náhodný vzorek, n =", len(rnd), "===")
print(classification_report(rnd["target"], rnd["pred_target_kw"], labels=TGT, zero_division=0))
print(pd.DataFrame(confusion_matrix(rnd["target"], rnd["pred_target_kw"], labels=TGT),
                   index=[f"gold_{s}" for s in TGT], columns=[f"pred_{s}" for s in TGT]))

# ---------- 4) Tabulka chyb pro ruční error analysis ----------
err = sub[sub["roberta_label"] != sub["target_sentiment"]].copy()
err["vader_also_wrong"] = err["vader_label"] != err["target_sentiment"]
err_cols = ["row_id", "sample_type", "target", "target_sentiment", "intensity", "roberta_label",
            "roberta_score", "vader_label", "vader_compound", "vader_also_wrong",
            "target_uncertain", "text_sentiment"]
err[err_cols].sort_values("intensity", ascending=False).to_csv(
    f"{BASE}/data/sample/errors_roberta.csv", index=False)
print(f"\nChyb RoBERTy: {len(err)} z {len(sub)} -> uloženo do errors_roberta.csv")

all_gold.to_csv(f"{BASE}/data/sample/all_gold_predictions.csv", index=False)
