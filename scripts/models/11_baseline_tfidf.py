# ============================================================
# BLOK 10: BASELINY PRO SENTIMENT VŮČI CÍLI
# Co dělá:
#   Přidá k VADER a RoBERTě dva triviální baseliny, aby bylo vidět,
#   zda modely vůbec něco přidávají:
#     B0 "vždy negativní"  = vždy předpoví negative
#     B1 "podle cíle"      = pravidlo určené předem (ne naučené z dat):
#                            Rusko -> negative, Ukrajina -> positive
#     B2 "podle regexu"    = totéž, ale cíl se určí z textu klíčovými
#                            slovy (realistická varianta bez gold cíle)
#   Pro každý model spočítá accuracy a macro-F1 s 95% bootstrap
#   intervalem a recall po třídách, zvlášť pro náhodný vzorek
#   (hlavní výsledek) a pro spojené vzorky. Nakonec spárovaný
#   bootstrap: je RoBERTa lepší než baseline "podle cíle"?
# Vstup:  data/sample/all_gold_predictions.csv
# Výstup: data/sample/baselines_table.csv
# Pozor:  nepotřebuje GPU ani objekty z předchozích bloků.
#   B1 a B2 používají gold cíl jen k výběru komentářů (stejná
#   množina jako u VADER/RoBERTy, n = 243 / 110); do reportu
#   uvést jako omezení.
# ============================================================
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from google.colab import drive
drive.mount('/content/drive')

BASE = "/content/drive/MyDrive/ukraine_project"
SENT = ["negative", "neutral", "positive"]
rng = np.random.default_rng(42)
N_BOOT = 1000

df = pd.read_csv(f"{BASE}/data/sample/all_gold_predictions.csv")
sub = df[df["target"].isin(["ukraine", "russia"])].copy()
print("Komentářů s cílem Ukrajina/Rusko:", len(sub))
print(sub["sample_type"].value_counts().to_dict())

# ---------- Baseliny ----------
sub["B0_vzdy_negativni"] = "negative"
sub["B1_podle_gold_cile"] = sub["target"].map({"russia": "negative", "ukraine": "positive"})

UKR = r"ukrain|kyiv|kiev|zelensk|kharkiv|kherson|donbas|mariupol|bakhmut|azov|avdiivka|kramatorsk"
RUS = r"russia|putin|moscow|kremlin|ruzzia|vatnik|wagner|kursk|rosgvard|\bru\b"

def regex_baseline(text):
    u = bool(re.search(UKR, text, flags=re.I))
    r = bool(re.search(RUS, text, flags=re.I))
    if u and not r:
        return "positive"      # zmiňuje jen Ukrajinu
    return "negative"          # jen Rusko, obojí nebo nic

sub["B2_podle_regexu"] = sub["text_sentiment"].apply(regex_baseline)

MODELS = {
    "B0 vždy negativní": "B0_vzdy_negativni",
    "B1 podle gold cíle": "B1_podle_gold_cile",
    "B2 podle regexu": "B2_podle_regexu",
    "VADER": "vader_label",
    "RoBERTa": "roberta_label",
}

def acc(d, col): return accuracy_score(d["target_sentiment"], d[col])
def mf1(d, col):
    return f1_score(d["target_sentiment"], d[col], labels=SENT, average="macro", zero_division=0)

def ci(fn, d, col):
    vals = [fn(d.iloc[rng.integers(0, len(d), len(d))], col) for _ in range(N_BOOT)]
    return np.percentile(vals, [2.5, 97.5])

sets = {"random (hlavní)": sub[sub["sample_type"] == "random"], "vše spojené": sub}

rows = []
for sname, d in sets.items():
    for mname, col in MODELS.items():
        a_lo, a_hi = ci(acc, d, col)
        f_lo, f_hi = ci(mf1, d, col)
        rec = recall_score(d["target_sentiment"], d[col], labels=SENT,
                           average=None, zero_division=0)
        rows.append({
            "vzorek": sname, "n": len(d), "model": mname,
            "accuracy": round(acc(d, col), 3), "acc 95% CI": f"{a_lo:.2f}-{a_hi:.2f}",
            "macro-F1": round(mf1(d, col), 3), "F1 95% CI": f"{f_lo:.2f}-{f_hi:.2f}",
            "recall neg": round(rec[0], 2), "recall neu": round(rec[1], 2),
            "recall pos": round(rec[2], 2),
        })
res = pd.DataFrame(rows)
print("\n=== SENTIMENT VŮČI CÍLI: MODELY × BASELINY ===")
print(res.to_string(index=False))

# ---------- Rozložení gold labelů podle cíle (proč B1 funguje) ----------
print("\n=== Gold sentiment podle cíle (náhodný vzorek) ===")
print(pd.crosstab(sets["random (hlavní)"]["target"],
                  sets["random (hlavní)"]["target_sentiment"]))

# ---------- Spárovaný bootstrap ----------
print("\n=== Spárovaný bootstrap (rozdíl accuracy / macro-F1) ===")
def paired(d, col_a, col_b):
    da, df_ = [], []
    for _ in range(N_BOOT):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        da.append(acc(s, col_a) - acc(s, col_b))
        df_.append(mf1(s, col_a) - mf1(s, col_b))
    return (acc(d, col_a) - acc(d, col_b), np.percentile(da, [2.5, 97.5]),
            mf1(d, col_a) - mf1(d, col_b), np.percentile(df_, [2.5, 97.5]))

for sname, d in sets.items():
    for a, b in [("roberta_label", "B1_podle_gold_cile"),
                 ("roberta_label", "B0_vzdy_negativni"),
                 ("B2_podle_regexu", "roberta_label")]:
        da, cia, dfm, cif = paired(d, a, b)
        print(f"{sname:16s} {a} - {b}: acc {da:+.3f} [{cia[0]:+.3f}, {cia[1]:+.3f}] | "
              f"F1 {dfm:+.3f} [{cif[0]:+.3f}, {cif[1]:+.3f}]")

res.to_csv(f"{BASE}/data/sample/baselines_table.csv", index=False)
print("\nUloženo: baselines_table.csv")
