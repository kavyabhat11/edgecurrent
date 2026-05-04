"""Compare EAP train-size sweep results across runs.

Usage:
    python baselines_examples/compare_train_sweep.py results/ioi-t1-eap-test-train20 results/ioi-t1-eap-test-train50 results/ioi-t1-eap-test-train100 results/ioi-t1-eap-test

Prints:
    - KL/logit-diff per sparsity, side-by-side
    - Top-k edge set overlap between runs (Jaccard) at the kept-edges count
"""

import argparse
import csv
import json
import os


def load_metrics(out_dir):
    csv_path = None
    for f in os.listdir(out_dir):
        if f.endswith("_metrics.csv"):
            csv_path = os.path.join(out_dir, f)
            break
    if csv_path is None:
        raise FileNotFoundError(f"No *_metrics.csv in {out_dir}")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return rows


def load_edges(out_dir, n):
    """Return the set of (from, to) for the top-n edges by abs score."""
    path = os.path.join(out_dir, "eap_all_edges.json")
    with open(path) as f:
        edges = json.load(f)
    return {(e["from"], e["to"]) for e in edges[:n]}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="Result directories to compare")
    p.add_argument("--metric-key", default="kl_model_circuit",
                   help="Which CSV column to display (kl_model_circuit, logit_difference, prob_diff, ...)")
    args = p.parse_args()

    # Load metrics for each run
    runs = {}
    for d in args.dirs:
        label = os.path.basename(d.rstrip("/"))
        runs[label] = load_metrics(d)

    labels = list(runs.keys())

    # 1) Side-by-side metric table
    print(f"\n=== {args.metric_key} per sparsity ===")
    sparsities = [r["sparsity_percent"] for r in runs[labels[0]]]
    header = f"{'sparsity':>10}  {'edges':>8}  " + "  ".join(f"{l:>30}" for l in labels)
    print(header)
    for i, sp in enumerate(sparsities):
        edges = runs[labels[0]][i]["num_edges_kept"]
        vals = "  ".join(f"{float(runs[l][i][args.metric_key]):>30.6f}" for l in labels)
        print(f"{sp:>10}  {edges:>8}  {vals}")

    # 2) Pairwise Jaccard of top-k edges at each sparsity
    print(f"\n=== Top-k edge-set overlap (Jaccard) between runs ===")
    print("If 1.0 across the board, every run picked the SAME circuit.")
    print(f"{'sparsity':>10}  {'edges':>8}  " + "  ".join(f"{l1[:14]} vs {l2[:14]}"
        for i, l1 in enumerate(labels) for l2 in labels[i+1:]))

    for i, sp in enumerate(sparsities):
        edges_n = int(runs[labels[0]][i]["num_edges_kept"])
        sets = {l: load_edges(d, edges_n) for l, d in zip(labels, args.dirs)}
        pairs = []
        for j, l1 in enumerate(labels):
            for l2 in labels[j+1:]:
                pairs.append(jaccard(sets[l1], sets[l2]))
        pairs_str = "  ".join(f"{p:>30.4f}" for p in pairs)
        print(f"{sp:>10}  {edges_n:>8}  {pairs_str}")


if __name__ == "__main__":
    main()
