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
    p.add_argument("--dataset-path", default="data/data/datasets/ioi/")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--train-examples", type=int, default=200)
    p.add_argument("--eval-examples", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--sparsities", default="0,93,94,95,96,96.5,96.6,97,97.5,98,98.5,99,100")
    p.add_argument("--out-dir", default="results/ioi-eap-paper-kl")
    return p.parse_args()


def tokenize_pair(clean_texts, corr_texts, tokenizer):
    all_texts = clean_texts + corr_texts
    tmp = tokenizer(all_texts, return_tensors="pt", padding=True)
    max_len = tmp.input_ids.shape[1]

    clean = tokenizer(clean_texts, return_tensors="pt", padding="max_length", max_length=max_len)
    corr = tokenizer(corr_texts, return_tensors="pt", padding="max_length", max_length=max_len)

    return clean.input_ids, corr.input_ids, clean.attention_mask


def get_ioi_data(ds, tokenizer, max_examples):
    if max_examples is not None and max_examples < len(ds):
        ds = ds.select(range(max_examples))

    clean_texts = list(ds["ioi_sentences"])
    corr_texts = list(ds["corr_ioi_sentences"])

    tokens, corr_tokens, attn = tokenize_pair(clean_texts, corr_texts, tokenizer)

    pred_indices = []
    correct_tokens = []
    distractor_tokens = []

    for i in range(len(ds)):
        last_real_pos = int(attn[i].sum().item()) - 1
        pred_pos = last_real_pos - 1

        correct_tok = int(tokens[i, last_real_pos].item())
        distractor_tok = int(tokenizer.encode(" " + ds[i]["b"])[-1])

        pred_indices.append(pred_pos)
        correct_tokens.append(correct_tok)
        distractor_tokens.append(distractor_tok)

    return (
        tokens,
        corr_tokens,
        torch.LongTensor(pred_indices),
        torch.LongTensor(correct_tokens),
        torch.LongTensor(distractor_tokens),
    )


def gather_at_positions(logits, indices):
    return torch.gather(
        logits,
        1,
        indices.reshape(-1, 1, 1).repeat(1, 1, logits.shape[-1]),
    ).squeeze(1)


def logprob_diff(logits, indices, correct, distractor):
    logits_at_pos = gather_at_positions(logits, indices)
    log_probs = F.log_softmax(logits_at_pos, dim=-1)

    correct_lp = torch.gather(log_probs, 1, correct.reshape(-1, 1)).squeeze(1)
    distractor_lp = torch.gather(log_probs, 1, distractor.reshape(-1, 1)).squeeze(1)

    return (correct_lp - distractor_lp).mean()


def kl_model_circuit(circuit_logits, full_logits, indices):
    circuit_at_pos = gather_at_positions(circuit_logits, indices)
    full_at_pos = gather_at_positions(full_logits, indices)

    circuit_log_probs = F.log_softmax(circuit_at_pos, dim=-1)
    full_log_probs = F.log_softmax(full_at_pos, dim=-1)

    return F.kl_div(
        circuit_log_probs,
        full_log_probs,
        log_target=True,
        reduction="batchmean",
    )


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

    # 1 = removed/corrupted edge
    for hook_name, down_slice in graph.downstream_hook_slice.items():
        fake_hook = type("FakeHook", (), {})()
        fake_hook.name = hook_name
        fake_hook.layer = lambda h=hook_name: int(h.split(".")[1])

        up_slice = graph.get_slice_previous_upstream_nodes(fake_hook)
        adj[up_slice, down_slice] = 1.0

    # 0 = kept/clean edge
    for src, dst, _score in keep_edges:
        if src in graph.upstream_node_index and dst in graph.downstream_node_index:
            adj[graph.upstream_node_index[src], graph.downstream_node_index[dst]] = 0.0

    return adj


@torch.no_grad()
def get_full_model_logits(model, tokens, batch_size):
    outs = []

    for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Full model logits"):
        batch = tokens[i:i + batch_size].to(model.cfg.device)
        outs.append(model(batch, return_type="logits").cpu())

    return torch.cat(outs, dim=0)


def circuit_logits_for_edges(model, clean_tokens, corr_tokens, keep_edges, batch_size):
    graph = EAPGraph(
        model.cfg,
        upstream_nodes=["resid_pre", "head", "mlp"],
        downstream_nodes=["head", "mlp", "resid_post"],
    )

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

    corr_hook = partial(
        EAP_corrupted_forward_hook,
        upstream_activations_difference=upstream_diff,
        graph=graph,
    )

    clean_hook = partial(
        EAP_clean_forward_hook,
        upstream_activations_difference=upstream_diff,
        graph=graph,
    )

    patch_hook = partial(
        EAP_downstream_patching_hook,
        upstream_activations_difference=upstream_diff,
        graph=graph,
    )

    for i in tqdm(range(0, num_prompts, batch_size), desc="Circuit eval"):
        batch_clean = clean_tokens[i:i + batch_size].to(model.cfg.device)
        batch_corr = corr_tokens[i:i + batch_size].to(model.cfg.device)

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
            logits = model(batch_clean, return_type="logits").cpu()

        logits_out.append(logits)
        model.reset_hooks()

    out = torch.cat(logits_out, dim=0)
    print("DEBUG circuit_logits shape:", out.shape)
    return out


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

    dataset = load_from_disk(args.dataset_path)

    train_ds = dataset[args.train_split]
    eval_ds = dataset[args.eval_split]

    train_toks, train_corr_toks, train_idx, train_correct, train_distractor = get_ioi_data(
        train_ds,
        tokenizer,
        args.train_examples,
    )

    usable_train_n = (train_toks.shape[0] // args.batch_size) * args.batch_size

    train_toks = train_toks[:usable_train_n]
    train_corr_toks = train_corr_toks[:usable_train_n]
    train_idx = train_idx[:usable_train_n]
    train_correct = train_correct[:usable_train_n]
    train_distractor = train_distractor[:usable_train_n]

    print("Computing full model logits for KL-based EAP metric...")
    train_full_logits = get_full_model_logits(model, train_toks, args.batch_size)

    metric_state = {"start": 0}

    def eap_kl_metric(logits):
        start = metric_state["start"]
        batch_n = logits.shape[0]
        end = start + batch_n
        metric_state["start"] = end

        indices = train_idx[start:end].to(logits.device)
        target_logits = train_full_logits[start:end].to(logits.device)

        kl = kl_model_circuit(logits, target_logits, indices)

        # EAP ranks edges by positive contribution to the objective.
        # Since lower KL is better, use negative KL.
        return -kl

    print("Running KL-based EAP scoring...")

    graph = EAP(
        model=model,
        clean_tokens=train_toks,
        corrupted_tokens=train_corr_toks,
        metric=eap_kl_metric,
        upstream_nodes=["resid_pre", "head", "mlp"],
        downstream_nodes=["head", "mlp", "resid_post"],
        batch_size=args.batch_size,
    )

    total_valid_edges = count_valid_edges(graph)
    print("Total valid edges:", total_valid_edges)

    all_edges = graph.top_edges(
        n=graph.eap_scores.numel(),
        abs_scores=False,
    )

    scores_path = os.path.join(args.out_dir, "eap_all_edges_kl_metric.json")

    with open(scores_path, "w") as f:
        json.dump(
            [{"from": s, "to": t, "score": float(v)} for s, t, v in all_edges],
            f,
            indent=2,
        )

    print("Saved KL-based EAP ranking to", scores_path)

    eval_toks, eval_corr_toks, eval_idx, eval_correct, eval_distractor = get_ioi_data(
        eval_ds,
        tokenizer,
        args.eval_examples,
    )

    usable_eval_n = (eval_toks.shape[0] // args.eval_batch_size) * args.eval_batch_size

    eval_toks = eval_toks[:usable_eval_n]
    eval_corr_toks = eval_corr_toks[:usable_eval_n]
    eval_idx = eval_idx[:usable_eval_n]
    eval_correct = eval_correct[:usable_eval_n]
    eval_distractor = eval_distractor[:usable_eval_n]

    print("DEBUG eval_toks shape:", eval_toks.shape)

    full_logits = get_full_model_logits(model, eval_toks, args.eval_batch_size)

    full_ld = logprob_diff(
        full_logits,
        eval_idx,
        eval_correct,
        eval_distractor,
    ).item()

    rows = []
    sparsities = [float(x) for x in args.sparsities.split(",")]

    for sparsity in sparsities:
        k = int(round((1.0 - sparsity / 100.0) * total_valid_edges))
        k = max(1, min(k, len(all_edges)))

        keep_edges = all_edges[:k]

        circuit_logits = circuit_logits_for_edges(
            model,
            eval_toks,
            eval_corr_toks,
            keep_edges,
            args.eval_batch_size,
        )

        kl = kl_model_circuit(
            circuit_logits,
            full_logits[:circuit_logits.shape[0]],
            eval_idx[:circuit_logits.shape[0]],
        ).item()

        ld = logprob_diff(
            circuit_logits,
            eval_idx[:circuit_logits.shape[0]],
            eval_correct[:circuit_logits.shape[0]],
            eval_distractor[:circuit_logits.shape[0]],
        ).item()

        row = {
            "sparsity_percent": sparsity,
            "num_edges_kept": k,
            "kl_model_circuit": kl,
            "logit_difference": ld,
            "full_model_logit_difference": full_ld,
            "eval_examples": int(circuit_logits.shape[0]),
        }

        rows.append(row)
        print(row)

    csv_path = os.path.join(args.out_dir, f"eap_ioi_{args.eval_split}_metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved metrics to", csv_path)


if __name__ == "__main__":
    main()