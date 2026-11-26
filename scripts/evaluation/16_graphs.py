"""
Step 15: figures for the final report, saved to figures/ (PNG, 300 dpi).
The stance figures use the AI labels of the 1900 sampled comments.
The model figures use the results in data/results/ (run 15_compare.py first).
"""
import sys
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import COUNTRIES, FIGURES, LABELS, N_BOOT, RESULTS, SEED
from utils import load_pairs

# The dates show when the middle 90 % of the sampled comments of each event were written
EVENTS = {
    "1_invasion": "Invasion\n10–18 Mar 2022",
    "2_bucha": "Bucha\n11–24 Apr 2022",
    "3_nordstream": "Nord Stream\n3–18 Oct 2022",
    "4_grain": "Grain dispute\n15 Sep–7 Oct 2023",
    "5_carlson_aid": "Carlson & US aid\n16 Feb–2 Mar 2024",
}
GROUPS = {"worldnews": ("r/worldnews", "#2a78d6"), "europe": ("r/europe", "#eb6834"),
          "UkraineRussiaReport": ("r/UkraineRussiaReport", "#1baf7a"),
          "US": ("US politics subreddits", "#eda100")}
# Stance colors: blue and red poles, gray middle (bar color, color of the text inside)
STANCES = {"pro": ("#2a78d6", "white"), "neutral": ("#dddcd6", "#0b0b0b"),
           "against": ("#e34948", "white")}
COUNTRY_COLORS = {"russia": "#2a78d6", "ukraine": "#eb6834"}
MODELS = {"finetuned": "Fine-tuned RoBERTa", "sentiment_roberta": "Sentiment RoBERTa",
          "tfidf": "TF-IDF + log. regression", "vader": "VADER",
          "zeroshot": "Zero-shot NLI", "majority": "Majority class"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)
    print("saved", FIGURES / f"{name}.png")


def sample_composition(pairs):
    comments = pairs.drop_duplicates("row_id")
    table = pd.crosstab(comments.event, comments.sub_group).reindex(index=EVENTS, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.8, 4))
    bottom = np.zeros(len(table))
    for group, (name, color) in GROUPS.items():
        ax.bar(range(len(table)), table[group], bottom=bottom, color=color, width=0.65,
               label=name, edgecolor="white", linewidth=1.2)
        for x, (value, base) in enumerate(zip(table[group], bottom)):
            if value:
                ax.text(x, base + value / 2, str(value), ha="center", va="center", fontsize=8,
                        color=INK if group == "US" else "white")
        bottom += table[group].values
    ax.set_xticks(range(len(table)), EVENTS.values())
    ax.set_ylabel("comments")
    ax.set_title(f"Sample composition by event and subreddit (n = {len(comments)})", pad=28)
    ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(-0.02, 1.0), fontsize=8)
    save(fig, "1_sample_composition")


def stance_shares(pairs, rng):
    """Share of each stance per event and country, among comments about the country.
    The 95 % interval resamples whole threads, because comments in a thread are related."""
    rows = []
    about = pairs[pairs.label != "na"]
    for (country, event), part in about.groupby(["country", "event"]):
        counts = pd.crosstab(part.parent_post_id, part.label).reindex(columns=list(STANCES), fill_value=0).values
        samples = counts[rng.integers(0, len(counts), (N_BOOT, len(counts)))].sum(axis=1)
        boot = samples / samples.sum(axis=1, keepdims=True)
        low, high = np.percentile(boot, [2.5, 97.5], axis=0)
        for i, stance in enumerate(STANCES):
            rows.append({"country": country, "event": event, "stance": stance, "n": len(part),
                         "share": counts[:, i].sum() / len(part), "low": low[i], "high": high[i]})
    return pd.DataFrame(rows)


def stance_bars(shares, title, name):
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.8), sharex=True)
    for ax, (country, country_name) in zip(axes, COUNTRIES.items()):
        part = shares[shares.country == country]
        table = part.pivot(index="event", columns="stance", values="share").reindex(EVENTS)
        n = part.drop_duplicates("event").set_index("event").n.reindex(EVENTS)
        bottom = np.zeros(len(table))
        for stance, (color, text_color) in STANCES.items():
            values = table[stance].values * 100
            ax.bar(range(len(table)), values, bottom=bottom, color=color, width=0.65,
                   label=stance, edgecolor="white", linewidth=1.2)
            for x, (value, base) in enumerate(zip(values, bottom)):
                if value >= 6:
                    ax.text(x, base + value / 2, f"{value:.0f} %", ha="center", va="center",
                            fontsize=8, color=text_color)
            bottom += values
        for x, count in enumerate(n):
            ax.text(x, 102, f"n = {count}", ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 25))
        ax.set_ylabel("% of comments")
        ax.set_title(f"Stance towards {country_name}")
    axes[1].set_xticks(range(len(EVENTS)), EVENTS.values())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, -0.045, "Only comments about the given country (n above each bar). Labels were made by AI.\n"
             "Dates: period in which the middle 90 % of all sampled comments of the event were written.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    save(fig, name)


def stance_intervals(shares):
    """Pro and against shares with 95 % intervals. Events are separate windows, so no lines."""
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True, sharey=True)
    for ax, (country, country_name) in zip(axes, COUNTRIES.items()):
        for stance, offset, marker in [("pro", -0.12, "o"), ("against", 0.12, "s")]:
            part = shares[(shares.country == country) & (shares.stance == stance)]
            part = part.set_index("event").reindex(EVENTS)
            error = np.clip([part.share - part.low, part.high - part.share], 0, None) * 100
            ax.errorbar(np.arange(len(part)) + offset, part.share * 100, yerr=error, fmt=marker,
                        color=STANCES[stance][0], capsize=3, markersize=7, label=stance)
        ax.set_title(f"Stance towards {country_name}")
        ax.set_ylabel("% of comments")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(ncol=2, loc="upper right")
    axes[1].set_xticks(range(len(EVENTS)), EVENTS.values())
    fig.text(0.01, -0.01, "95 % bootstrap intervals over threads. Neutral comments make up the rest.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "4_stance_intervals")


def model_comparison(metrics):
    data = metrics[(metrics.split == "test") & (metrics.subset == "all")]
    order = [m for m in MODELS if m in set(data.model)]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    columns = [("accuracy", "acc_low", "acc_high", "Accuracy"), ("macro_f1", "f1_low", "f1_high", "Macro-F1")]
    for ax, (value, low, high, title) in zip(axes, columns):
        for offset, (country, name) in zip([-0.19, 0.19], COUNTRIES.items()):
            part = data[data.country == country].set_index("model").reindex(order)
            error = np.clip([part[value] - part[low], part[high] - part[value]], 0, None)
            ax.barh(y + offset, part[value], height=0.36, color=COUNTRY_COLORS[country], label=name,
                    xerr=error, error_kw={"ecolor": MUTED, "capsize": 2, "linewidth": 0.8})
            for pos, v, top in zip(y + offset, part[value], part[high]):
                ax.text(top + 0.015, pos, f"{v:.2f}", va="center", fontsize=7.5, color=MUTED)
        ax.set_xlim(0, 0.92)
        ax.set_title(title)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, [MODELS[m] for m in order])
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Model comparison on the test set (300 comments, 95 % intervals)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "5_model_comparison")


def confusion_matrices(pairs):
    predictions = pd.read_csv(RESULTS / "pred_finetuned.csv", keep_default_na=False)
    data = predictions.merge(pairs, on=["row_id", "country", "split"])
    data = data[data.split == "test"]
    blues = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#86b6ef", "#2a78d6", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for ax, (country, name) in zip(axes, COUNTRIES.items()):
        part = data[data.country == country]
        counts = confusion_matrix(part.label, part.prediction, labels=LABELS)
        share = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap=blues, vmin=0, vmax=1)
        for i, j in np.ndindex(counts.shape):
            ax.text(j, i, f"{counts[i, j]}\n{share[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if share[i, j] > 0.5 else INK)
        ax.set_xticks(range(len(LABELS)), LABELS)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("model prediction")
        ax.set_title(name)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("AI label")
    fig.suptitle("Fine-tuned RoBERTa: confusion matrices on the test set (count and % of row)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "6_confusion_matrix")


def learning_curve(metrics):
    runs = {"finetuned_25": 25, "finetuned_50": 50, "finetuned_75": 75, "finetuned": 100}
    data = metrics[(metrics.split == "test") & (metrics.subset == "all") & metrics.model.isin(runs)]
    data = data.assign(size=data.model.map(runs)).sort_values("size")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for offset, (country, name) in zip([-0.8, 0.8], COUNTRIES.items()):
        part = data[data.country == country]
        ax.errorbar(part["size"] + offset, part.macro_f1,
                    yerr=[part.macro_f1 - part.f1_low, part.f1_high - part.macro_f1],
                    marker="o", markersize=7, linewidth=2, capsize=3,
                    color=COUNTRY_COLORS[country], label=name)
    ax.set_xticks(list(runs.values()), [f"{s} %" for s in runs.values()])
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("share of the 1300 training comments")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Learning curve of the fine-tuned model")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    save(fig, "7_learning_curve")


def main():
    FIGURES.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    pairs = load_pairs()
    metrics = pd.read_csv(RESULTS / "metrics.csv")

    sample_composition(pairs)
    shares = stance_shares(pairs, rng)
    stance_bars(shares, "Stance by war event (all subreddits)", "2_stance_by_event")
    # Control: only the two subreddits present at every event, so the mix changes less
    control = stance_shares(pairs[pairs.sub_group.isin(["worldnews", "europe"])], rng)
    stance_bars(control, "Control: r/worldnews and r/europe only", "3_stance_control")
    stance_intervals(shares)
    model_comparison(metrics)
    confusion_matrices(pairs)
    learning_curve(metrics)

    # The numbers behind the stance figures
    table = pd.concat([shares.assign(subreddits="all"), control.assign(subreddits="worldnews+europe")])
    table.round(4).to_csv(FIGURES / "stance_by_event.csv", index=False)


if __name__ == "__main__":
    main()
