"""
Steps 10-13: fine-tune RoBERTa. Input: comment + country name, output: one of the four labels.
The best epoch is chosen on validation. A checkpoint is saved during training;
if stopped, run the same command again to continue.

  python scripts/models/14_finetune.py                    all training data
  python scripts/models/14_finetune.py --fraction 0.25    learning curve: 25 % of the training data
"""
import sys
from pathlib import Path
import argparse
import os

import torch
from torch.nn.functional import cross_entropy
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

sys.path.append(str(Path(__file__).resolve().parents[1]))  # folder with config.py and utils.py
from config import BASE_MODEL, CHECKPOINTS, COUNTRIES, LABELS, MODELS, SEED
from utils import load_pairs, macro_f1, prediction_file, save_predictions

EPOCHS = 4               # how many times the model sees the training data
BATCH = 8                # comments per training step
LEARNING_RATE = 2e-5
MAX_LENGTH = 256         # longer texts are cut
SAVE_EVERY = 50          # training steps between checkpoints


def take_fraction(train, fraction):
    """Random whole threads up to the wanted share (smaller sets are parts of larger ones)."""
    sizes = train.groupby("parent_post_id").size().sample(frac=1, random_state=SEED)
    threads = sizes.index[sizes.cumsum() <= round(fraction * sizes.sum())]
    return train[train.parent_post_id.isin(threads)]


def encode(tokenizer, rows):
    """Turn each (comment, country) pair into numbers the model understands."""
    countries = [COUNTRIES[country] for country in rows.country]
    return tokenizer(list(rows.text), countries, padding=True, truncation="only_first",
                     max_length=MAX_LENGTH, return_tensors="pt")


def predict(model, tokenizer, rows):
    model.eval()
    result = []
    for start in range(0, len(rows), 16):
        with torch.no_grad():
            logits = model(**encode(tokenizer, rows.iloc[start:start + 16])).logits
        result += [LABELS[i] for i in logits.argmax(dim=1).tolist()]
    return result


def validation_score(model, tokenizer, val):
    """Mean of the macro-F1 for Russia and for Ukraine."""
    scored = val.assign(prediction=predict(model, tokenizer, val))
    return sum(macro_f1(part.label, part.prediction) for _, part in scored.groupby("country")) / 2


def save_torch_safely(content, path):
    """Write to a temporary file first, so stopping never leaves half a file."""
    tmp = path.with_suffix(".tmp")
    torch.save(content, tmp)
    os.replace(tmp, path)


def train_model(model, tokenizer, train, val, folder):
    """Train the model and leave it with the weights of the best epoch."""
    targets = torch.tensor([LABELS.index(label) for label in train.label])
    # Rare labels get a higher weight (square root of the inverse frequency)
    counts = torch.bincount(targets, minlength=len(LABELS)).clamp(min=1).float()
    weights = (counts.sum() / counts).sqrt()
    weights = weights / weights.mean()

    steps_per_epoch = (len(train) + BATCH - 1) // BATCH
    total_steps = steps_per_epoch * EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_file, best_file = folder / "checkpoint.pt", folder / "best.pt"
    step, best_score = 0, -1.0
    if checkpoint_file.exists():      # continue an interrupted training
        state = torch.load(checkpoint_file, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, best_score = state["step"], state["best"]

    print(f"{len(train)} training pairs, step {step} of {total_steps}", flush=True)
    while step < total_steps:
        epoch, position = divmod(step, steps_per_epoch)
        # The order depends only on the epoch, so a continued run sees the same batches
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED + epoch))
        batch = order[position * BATCH:(position + 1) * BATCH]

        # One training step: predict, measure the error, adjust the model
        model.train()
        logits = model(**encode(tokenizer, train.iloc[batch.numpy()])).logits
        loss = cross_entropy(logits, targets[batch], weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1
        if step % 10 == 0:
            print(f"  step {step}/{total_steps} | loss {loss.item():.3f}", flush=True)

        epoch_finished = step % steps_per_epoch == 0
        if epoch_finished:
            score = validation_score(model, tokenizer, val)
            print(f"  epoch {step // steps_per_epoch}: validation macro-F1 {score:.3f}", flush=True)
            if score > best_score:
                best_score = score
                save_torch_safely(model.state_dict(), best_file)
        if epoch_finished or step % SAVE_EVERY == 0:
            save_torch_safely({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "step": step,
                               "best": best_score}, checkpoint_file)

    model.load_state_dict(torch.load(best_file, weights_only=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=1.0, help="share of the training data")
    fraction = parser.parse_args().fraction
    name = "finetuned" if fraction == 1 else f"finetuned_{round(fraction * 100)}"
    if prediction_file(name).exists():
        print(name, "already done, skipping")
        return

    torch.manual_seed(SEED)
    pairs = load_pairs()
    train = take_fraction(pairs[pairs.split == "train"], fraction).reset_index(drop=True)
    evaluation = pairs[pairs.split != "train"].reset_index(drop=True)
    val = evaluation[evaluation.split == "val"]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABELS))
    folder = CHECKPOINTS / name
    train_model(model, tokenizer, train, val, folder)

    model.save_pretrained(MODELS / name)
    tokenizer.save_pretrained(MODELS / name)
    save_predictions(name, evaluation, predict(model, tokenizer, evaluation))
    # Training is finished: remove the large temporary files made by this script
    for temporary in ["checkpoint.pt", "best.pt"]:
        (folder / temporary).unlink(missing_ok=True)
    print(f"{name} finished, model saved to {MODELS / name}")


if __name__ == "__main__":
    main()
