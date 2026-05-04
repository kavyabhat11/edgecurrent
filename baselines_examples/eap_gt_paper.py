import argparse
import csv
import json
import os
import random
from functools import partial

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_from_disk
from transformer_lens import HookedTransformer

from eap.eap_wrapper import (
    EAP,
    EAP_corrupted_forward_hook,
    EAP_clean_forward_hook,
    EAP_downstream_patching_hook,
)
from eap.eap_graph import EAPGraph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", default="data/data/datasets/gt/")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--train-examples", type=int, default=200)
    p.add_argument("--eval-examples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--sparsities", default="93,94,95,96,97,98,99")
    p.add_argument("--out-dir", default="results/gt-eap-paper-kl")
    return p.parse_args()


def tokenize_pair(clean_texts, corr_texts, tokenizer):
    all_texts = clean_texts + corr_texts
    tmp = tokenizer(all_texts, return_tensors="pt", padding=True)
    max_len = tmp.input_ids.shape[1]
    clean = tokenizer(clean_texts, return_tensors="pt", padding="max_length", max_length=max_len)
    corr = tokenizer(corr_texts, return_tensors="pt", padding="max_length", max_length=max_len)
    return clean.input_ids, corr.input_ids, clean.attention_mask


def get_gt_data(ds, tokenizer, max_examples):
    if max_examples is not None and max_examples < len(ds):
        ds = ds.select(range(max_examples))

    clean_texts = [ds[i]["prefix"] for i in range(len(ds))]
    corr_texts = [ds[i]["corr_prefix"] for i in range(len(ds))]
    digits = [int(ds[i]["digits"]) for i in range(len(ds))]

    tokens, corr_tokens, attn = tokenize_pair(clean_texts, corr_texts, tokenizer)

    pred_indices = []
    for i in range(len(ds)):
        last_real_pos = int(attn[i].sum().item()) - 1
        pred_indices.append(last_real_pos)

    return (
        tokens,
        corr_tokens,
        torch.LongTensor(pred_indices),
        torch.LongTensor(digits),
    )


def gather_at_positions(logits, indices):
    return torch.gather(
        logits,
        1,
        indices.reshape(-1, 1, 1).repeat(1, 1, logits.shape[-1]),
    ).squeeze(1)


def get_digit_token_ids(tokenizer, device):
    bos = tokenizer.bos_token_id

    def first_real(ids):
        return ids[1] if (ids and ids[0] == bos) else ids[0]

    return torch.LongTensor(
        [first_real(tokenizer.encode("{:02d}".format(i))) for i in range(100)]
    ).to(device)


def gt_prob_diff(logits, indices, digits, digit_token_ids):
    """Probability mass on digits > threshold minus mass on digits < threshold.
    Accepts logits as [B, seq, vocab] OR [B, vocab] (pre-gathered)."""
    if logits.dim() == 3:
        logits_at_pos = gather_at_positions(logits, indices)
    else:
        logits_at_pos = logits
    digit_logits = logits_at_pos[:, digit_token_ids]       # [B, 100]
    probs = F.softmax(digit_logits, dim=-1)

    out = 0.0
    for j in range(probs.shape[0]):
        d = int(digits[j].item())
        out = out + (probs[j, d + 1:].sum() - probs[j, :d].sum())
    return out / probs.shape[0]


def kl_model_circuit_gt(circuit_logits, full_logits, indices, digit_token_ids):
    """KL over the 100 digit tokens at the prediction position."""
    if circuit_logits.dim() == 3:
        c = gather_at_positions(circuit_logits, indices)[:, digit_token_ids]
    else:
        c = circuit_logits[:, digit_token_ids]
    if full_logits.dim() == 3:
        f = gather_at_positions(full_logits, indices)[:, digit_token_ids]
    else:
        f = full_logits[:, digit_token_ids]
    c_log = F.log_softmax(c, dim=-1)
    f_log = F.log_softmax(f, dim=-1)
    return F.kl_div(c_log, f_log, log_target=True, reduction="batchmean")


def count_valid_edges(graph):
    total = 0
    for down_node in graph.downstream_nodes:
        hook_name = None
        idx = graph.downstream_node_index[down_node]
        for h, sl in graph.downstream_hook_slice.items():
            if sl.start <= idx < sl.stop:
                hook_name = h
                break
        fake_hook = type("FakeHook", (), {})()
        fake_hook.name = hook_name
        fake_hook.layer = lambda h=hook_name: int(h.split(".")[1])
        total += graph.get_slice_previous_upstream_nodes(fake_hook).stop
    return total


def make_removed_edge_matrix(graph, keep_edges):
    adj = torch.zeros(
        (graph.n_upstream_nodes, graph.n_downstream_nodes),
        device=graph.cfg.device,
    )
    for hook_name, down_slice in graph.downstream_hook_slice.items():
        fake_hook = type("FakeHook", (), {})()
        fake_hook.name = hook_name
        fake_hook.layer = lambda h=hook_name: int(h.split(".")[1])
        up_slice = graph.get_slice_previous_upstream_nodes(fake_hook)
        adj[up_slice, down_slice] = 1.0
    for src, dst, _score in keep_edges:
        if src in graph.upstream_node_index and dst in graph.downstream_node_index:
            adj[graph.upstream_node_index[src], graph.downstream_node_index[dst]] = 0.0
    return adj


@torch.no_grad()
def get_full_model_logits_at_pos(model, tokens, pred_indices, batch_size):
    outs = []
    for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Full model logits"):
        batch = tokens[i:i + batch_size].to(model.cfg.device)
        idx = pred_indices[i:i + batch_size].to(model.cfg.device)
        logits = model(batch, return_type="logits")
        gathered = torch.gather(
            logits,
            1,
            idx.reshape(-1, 1, 1).repeat(1, 1, logits.shape[-1]),
        ).squeeze(1)
        outs.append(gathered.cpu())
    return torch.cat(outs, dim=0)


def circuit_logits_for_edges(model, clean_tokens, corr_tokens, pred_indices, keep_edges, batch_size):
    graph = EAPGraph(model.cfg, upstream_nodes=["head", "mlp"], downstream_nodes=["head", "mlp"])
    graph.adj_matrix = make_removed_edge_matrix(graph, keep_edges)

    num_prompts, seq_len = clean_tokens.shape
    logits_out = []

    upstream_diff = torch.zeros(
        (batch_size, seq_len, graph.n_upstream_nodes, model.cfg.d_model),
        device=model.cfg.device,
        dtype=model.cfg.dtype,
        requires_grad=False,
    )

    upstream_filter = lambda name: name.endswith(tuple(graph.upstream_hooks))
    downstream_filter = lambda name: name.endswith(tuple(graph.downstream_hooks))

    corr_hook = partial(EAP_corrupted_forward_hook, upstream_activations_difference=upstream_diff, graph=graph)
    clean_hook = partial(EAP_clean_forward_hook, upstream_activations_difference=upstream_diff, graph=graph)
    patch_hook = partial(EAP_downstream_patching_hook, upstream_activations_difference=upstream_diff, graph=graph)

    for i in tqdm(range(0, num_prompts, batch_size), desc="Circuit eval"):
        batch_clean = clean_tokens[i:i + batch_size].to(model.cfg.device)
        batch_corr = corr_tokens[i:i + batch_size].to(model.cfg.device)
        batch_idx = pred_indices[i:i + batch_size].to(model.cfg.device)
        if batch_clean.shape[0] != batch_size:
            continue
        model.reset_hooks()
        upstream_diff.zero_()
        model.add_hook(upstream_filter, corr_hook, "fwd")
        with torch.no_grad():
            model(batch_corr, return_type=None)
        model.reset_hooks()
        model.add_hook(upstream_filter, clean_hook, "fwd")
        model.add_hook(downstream_filter, patch_hook, "fwd")
        with torch.no_grad():
            logits = model(batch_clean, return_type="logits")
            gathered = torch.gather(
                logits,
                1,
                batch_idx.reshape(-1, 1, 1).repeat(1, 1, logits.shape[-1]),
            ).squeeze(1).cpu()
        logits_out.append(gathered)
        model.reset_hooks()

    return torch.cat(logits_out, dim=0)


def main():
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device=args.device,
    )
    model.set_use_hook_mlp_in(True)
    model.set_use_split_qkv_input(True)
    model.set_use_attn_result(True)

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    digit_token_ids_cpu = get_digit_token_ids(tokenizer, "cpu")

    dataset = load_from_disk(args.dataset_path)
    train_ds = dataset[args.train_split]
    eval_ds = dataset[args.eval_split]

    train_toks, train_corr_toks, train_idx, train_digits = get_gt_data(
        train_ds, tokenizer, args.train_examples,
    )

    usable_train_n = (train_toks.shape[0] // args.batch_size) * args.batch_size
    train_toks = train_toks[:usable_train_n]
    train_corr_toks = train_corr_toks[:usable_train_n]
    train_idx = train_idx[:usable_train_n]
    train_digits = train_digits[:usable_train_n]

    metric_state = {"start": 0}

    def eap_metric(logits):
        start = metric_state["start"]
        batch_n = logits.shape[0]
        end = start + batch_n
        metric_state["start"] = end
        indices = train_idx[start:end].to(logits.device)
        digits = train_digits[start:end].to(logits.device)
        digit_ids = get_digit_token_ids(tokenizer, logits.device)
        return gt_prob_diff(logits, indices, digits, digit_ids)

    print("Running EAP scoring...")
    graph = EAP(
        model=model,
        clean_tokens=train_toks,
        corrupted_tokens=train_corr_toks,
        metric=eap_metric,
        upstream_nodes=["head", "mlp"],
        downstream_nodes=["head", "mlp"],
        batch_size=args.batch_size,
    )

    total_valid_edges = count_valid_edges(graph)
    print("Total valid edges:", total_valid_edges)

    all_edges = graph.top_edges(n=graph.eap_scores.numel(), abs_scores=True)

    scores_path = os.path.join(args.out_dir, "eap_all_edges.json")
    with open(scores_path, "w") as f:
        json.dump(
            [{"from": s, "to": t, "score": float(v)} for s, t, v in all_edges],
            f, indent=2,
        )

    eval_toks, eval_corr_toks, eval_idx, eval_digits = get_gt_data(
        eval_ds, tokenizer, args.eval_examples,
    )

    usable_eval_n = (eval_toks.shape[0] // args.eval_batch_size) * args.eval_batch_size
    eval_toks = eval_toks[:usable_eval_n]
    eval_corr_toks = eval_corr_toks[:usable_eval_n]
    eval_idx = eval_idx[:usable_eval_n]
    eval_digits = eval_digits[:usable_eval_n]

    print("DEBUG eval_toks shape:", eval_toks.shape)

    full_logits = get_full_model_logits_at_pos(model, eval_toks, eval_idx, args.eval_batch_size)
    full_pd = gt_prob_diff(full_logits, eval_idx, eval_digits, digit_token_ids_cpu).item()

    rows = []
    sparsities = [float(x) for x in args.sparsities.split(",")]

    for sparsity in sparsities:
        k = int(round((1.0 - sparsity / 100.0) * total_valid_edges))
        k = max(1, min(k, len(all_edges)))
        keep_edges = all_edges[:k]

        circuit_logits = circuit_logits_for_edges(
            model, eval_toks, eval_corr_toks, eval_idx, keep_edges, args.eval_batch_size,
        )

        kl = kl_model_circuit_gt(
            circuit_logits,
            full_logits[:circuit_logits.shape[0]],
            eval_idx[:circuit_logits.shape[0]],
            digit_token_ids_cpu,
        ).item()

        pd = gt_prob_diff(
            circuit_logits,
            eval_idx[:circuit_logits.shape[0]],
            eval_digits[:circuit_logits.shape[0]],
            digit_token_ids_cpu,
        ).item()

        row = {
            "sparsity_percent": sparsity,
            "num_edges_kept": k,
            "kl_model_circuit": kl,
            "prob_diff": pd,
            "full_model_prob_diff": full_pd,
            "eval_examples": int(circuit_logits.shape[0]),
        }
        rows.append(row)
        print(row)

    csv_path = os.path.join(args.out_dir, f"eap_gt_{args.eval_split}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved metrics to", csv_path)


if __name__ == "__main__":
    main()
