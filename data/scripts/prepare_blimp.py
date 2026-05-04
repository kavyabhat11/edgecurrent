"""Download and prepare a BLiMP subset for Edge Pruning / EAP baselines.

Saves a HuggingFace DatasetDict to data/datasets/blimp_<subset>/ with
train/validation/test splits. Each example has fields:

    sentence_good   : grammatical full sentence
    sentence_bad    : ungrammatical minimal pair
    target_word     : the differing token in sentence_good (e.g., "are")
    distractor_word : the differing token in sentence_bad (e.g., "is")
    prefix          : shared prefix up to (but not including) the differing token

We restrict to:
  - one_prefix_method=True (there is a shared prefix)
  - target_word and distractor_word both single-token under GPT-2's tokenizer

Default subset: regular_plural_subject_verb_agreement_1 -- the cleanest
case of subject-verb agreement (clear single-token verb at known position).

Usage:
    python data/scripts/prepare_blimp.py
    python data/scripts/prepare_blimp.py --subset distractor_agreement_relative_clause
"""

import argparse
import os
import random

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


def first_diff_word(good: str, bad: str):
    g_tokens = good.split()
    b_tokens = bad.split()
    for i, (gt, bt) in enumerate(zip(g_tokens, b_tokens)):
        if gt != bt:
            prefix = " ".join(g_tokens[:i])
            return prefix, gt, bt, i
    return None, None, None, None


def filter_examples(raw, tokenizer):
    rows = []
    for ex in raw:
        if not ex.get("one_prefix_method", False):
            continue
        prefix, target, distractor, idx = first_diff_word(
            ex["sentence_good"], ex["sentence_bad"]
        )
        if prefix is None:
            continue
        # Both candidate words must be single-token for GPT-2's BPE.
        # GPT-2 tokenizes leading-space-prefixed tokens differently; use the
        # leading-space form since that matches how the model would see the
        # word mid-sentence.
        t_ids = tokenizer.encode(" " + target, add_special_tokens=False)
        d_ids = tokenizer.encode(" " + distractor, add_special_tokens=False)
        if len(t_ids) != 1 or len(d_ids) != 1:
            continue
        rows.append({
            "sentence_good": ex["sentence_good"],
            "sentence_bad": ex["sentence_bad"],
            "target_word": target,
            "distractor_word": distractor,
            "prefix": prefix,
            "diff_word_index": idx,
            "UID": ex.get("UID", ""),
            "linguistics_term": ex.get("linguistics_term", ""),
        })
    return rows


def split_rows(rows, seed=42, train_n=200, val_n=200):
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    train = rows[:train_n]
    val = rows[train_n : train_n + val_n]
    test = rows[train_n + val_n :]
    return train, val, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="regular_plural_subject_verb_agreement_1",
                   help="BLiMP config name. Examples: regular_plural_subject_verb_agreement_1, "
                        "irregular_plural_subject_verb_agreement_1, "
                        "distractor_agreement_relative_clause, etc.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory. Default: data/datasets/blimp_<subset>/")
    p.add_argument("--tokenizer", default="gpt2",
                   help="Tokenizer to use for single-token filtering (matches GPT-2).")
    p.add_argument("--train-n", type=int, default=200)
    p.add_argument("--val-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = f"data/datasets/blimp_{args.subset}/"

    print(f"Loading BLiMP subset {args.subset}...")
    ds = load_dataset("nyu-mll/blimp", args.subset)
    raw = ds["train"]  # BLiMP has only 'train' split (1000 examples)
    print(f"  {len(raw)} raw minimal pairs")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Filtering to one_prefix + single-token target/distractor...")
    rows = filter_examples(raw, tokenizer)
    print(f"  {len(rows)} usable pairs after filtering")

    if len(rows) < args.train_n + args.val_n + 10:
        raise RuntimeError(
            f"Too few usable pairs ({len(rows)}) to make train={args.train_n} + "
            f"val={args.val_n} + nontrivial test split. Lower --train-n/--val-n or pick another subset."
        )

    train, val, test = split_rows(rows, seed=args.seed, train_n=args.train_n, val_n=args.val_n)
    print(f"Splits: train={len(train)}  val={len(val)}  test={len(test)}")

    dd = DatasetDict({
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    })

    os.makedirs(args.out_dir, exist_ok=True)
    dd.save_to_disk(args.out_dir)
    print(f"Saved to {args.out_dir}")

    print("\n--- Sample (first row of train) ---")
    print(train[0])


if __name__ == "__main__":
    main()
