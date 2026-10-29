#!/usr/bin/env python3
"""World 2 steering pipeline: SVD direction from task_0, evaluated on task_0.

Stages (each skipped if output exists):
  1. Generate 1000 extraction samples on task_0 across all domains.
  2. Mean-pool layer-6 hidden states for the aligned and misaligned models.
  3. SVD top right singular vector of (mis - al). Sign-align to point from
     aligned toward misaligned. Record the natural scale alpha.
  4. Eval on the task_0 rows of the eval_full val set: baseline,
     aligned + m*alpha*v, and misaligned - m*alpha*v.

Outputs go to sweep/steering_task0/{data,artifacts,eval}.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data_gen_v3.generator import generate_dataset, save_dataset
from data_gen_v3.parser import TokenParser
from data_gen_v3.world import World
from pretraining_v3.data import load_val_examples_v3
from pretraining_v3.model import load_model, unwrap_model

EXP = REPO / "experiments/world2"
SWEEP = EXP / "sweep"
STEERING = SWEEP / "steering_task0"
WORLD_PATH = EXP / "world.json"
ALIGNED_MODEL = SWEEP / "cells/pretrain_15000/align_3000/exp_out/final_model/model.pt"
MISALIGNED_MODEL = SWEEP / "cells/pretrain_15000/align_3000/misalign_8/exp_out/final_model/model.pt"
VAL_JSONL = EXP / "pretrain/align/eval_full/data/val.jsonl"

TASK_FILTER = "task_0"

MODEL_NAME = "openai-community/gpt2"
SEQ_LEN = 256
N_LAYERS = 12
MID_LAYER = N_LAYERS // 2
HIDDEN_DIM = 768

N_EXTRACT = 1000
EXTRACT_SEED = 8888
BATCH_SIZE = 32
MAX_NEW_TOKENS = 25
POS_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
NEG_MULTIPLIERS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Stage 1: Data generation — task_0 only, all domains
# ---------------------------------------------------------------------------

def stage_generate_data():
    out_path = STEERING / "data/extraction_1k.jsonl"
    if out_path.exists():
        print(f"SKIP stage 1: {out_path} exists")
        return
    print(f"Stage 1: Generating extraction data ({TASK_FILTER} only) ...")
    world = World.load(str(WORLD_PATH))
    examples = generate_dataset(
        world=world,
        world_path=str(WORLD_PATH),
        domain_weights={f"domain_{i}": 1.0 for i in range(len(world.domains))},
        task_weights={TASK_FILTER: 1.0},
        variant_weights={1: 1.0},
        num_samples=N_EXTRACT,
        seed=EXTRACT_SEED,
        num_workers=4,
    )
    save_dataset(examples, str(out_path))
    print(f"  Saved {len(examples)} examples (all {TASK_FILTER})")


# ---------------------------------------------------------------------------
# Stage 2: Hidden state extraction
# ---------------------------------------------------------------------------

def _load_weights(model, path, device):
    state = torch.load(str(path), map_location=device, weights_only=True)
    model.load_state_dict(state)


def extract_hidden_states(model, examples, vocab, device, layer_idx, batch_size):
    model.eval()
    raw = unwrap_model(model)
    captured = {}

    def hook_fn(module, inp, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()

    handle = raw.transformer.h[layer_idx].register_forward_hook(hook_fn)
    all_hidden = []

    with torch.no_grad():
        for b0 in range(0, len(examples), batch_size):
            batch = examples[b0 : b0 + batch_size]
            bs = len(batch)
            prompt_ids_list = [ex["prompt_ids"] for ex in batch]
            prompt_lens = [len(p) for p in prompt_ids_list]
            max_plen = max(prompt_lens)

            input_ids = torch.full((bs, max_plen), vocab.pad_token_id, dtype=torch.long, device=device)
            attn = torch.zeros(bs, max_plen, dtype=torch.long, device=device)
            for i, p in enumerate(prompt_ids_list):
                pad = max_plen - len(p)
                input_ids[i, pad:] = torch.tensor(p, dtype=torch.long)
                attn[i, pad:] = 1

            gen_out = model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=vocab.pad_token_id, eos_token_id=vocab.end_token_id,
            )
            total_len = gen_out.shape[1]

            full_mask = torch.zeros(bs, total_len, dtype=torch.long, device=device)
            for i in range(bs):
                p_pad = max_plen - prompt_lens[i]
                full_mask[i, p_pad:max_plen] = 1
                for j in range(max_plen, total_len):
                    tok = gen_out[i, j].item()
                    if tok == vocab.pad_token_id:
                        break
                    full_mask[i, j] = 1
                    if tok == vocab.end_token_id:
                        break

            model(input_ids=gen_out, attention_mask=full_mask)
            hs = captured["hs"]

            for i in range(bs):
                mask_i = full_mask[i].bool()
                n_tok = mask_i.sum().item()
                if n_tok == 0:
                    all_hidden.append(torch.zeros(HIDDEN_DIM))
                    continue
                all_hidden.append(hs[i, mask_i].mean(dim=0).cpu().float())

            done = b0 + bs
            if done % 200 < batch_size:
                print(f"    {done}/{len(examples)}")

    handle.remove()
    return torch.stack(all_hidden)


def stage_extract():
    a_out = STEERING / "artifacts/aligned_hidden.pt"
    m_out = STEERING / "artifacts/misaligned_hidden.pt"
    if a_out.exists() and m_out.exists():
        print("SKIP stage 2: hidden states exist")
        return

    print("Stage 2: Extracting hidden states ...")
    world = World.load(str(WORLD_PATH))
    vocab = world.vocab
    examples = load_val_examples_v3(
        str(STEERING / "data/extraction_1k.jsonl"), vocab, max_samples=0
    )
    print(f"  {len(examples)} extraction examples")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, mpath, opath in [
        ("aligned", ALIGNED_MODEL, a_out),
        ("misaligned", MISALIGNED_MODEL, m_out),
    ]:
        if opath.exists():
            print(f"  SKIP {name}")
            continue
        print(f"  Extracting: {name}")
        model = load_model(MODEL_NAME, vocab, SEQ_LEN, device, use_pretrained_weights=False)
        _load_weights(model, mpath, device)
        h = extract_hidden_states(model, examples, vocab, device, MID_LAYER, BATCH_SIZE)
        torch.save(h, str(opath))
        print(f"  {name} -> {h.shape}")
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 3: SVD direction
# ---------------------------------------------------------------------------

def stage_svd():
    v1_path = STEERING / "artifacts/v1.pt"
    if v1_path.exists():
        print("SKIP stage 3: SVD exists")
        return

    print("Stage 3: SVD ...")
    ah = torch.load(str(STEERING / "artifacts/aligned_hidden.pt"), weights_only=True)
    mh = torch.load(str(STEERING / "artifacts/misaligned_hidden.pt"), weights_only=True)

    D = mh - ah
    print(f"  D shape {D.shape}, mean ||d||={D.norm(dim=1).mean():.4f}")

    _, S, Vt = torch.linalg.svd(D, full_matrices=False)
    v1 = Vt[0]

    if v1 @ D.mean(0) < 0:
        v1 = -v1
        print("  Flipped v1 sign")

    alpha = (D @ v1).mean().item()
    total_var = (S**2).sum().item()
    evr = [(S[i].item() ** 2) / total_var for i in range(min(20, len(S)))]

    print(f"  alpha={alpha:.4f}  v1 explains {evr[0]*100:.1f}%")

    torch.save(v1, str(v1_path))
    torch.save(S, str(STEERING / "artifacts/singular_values.pt"))
    info = {
        "alpha": alpha,
        "v1_norm": v1.norm().item(),
        "n_samples": int(D.shape[0]),
        "hidden_dim": int(D.shape[1]),
        "layer": MID_LAYER,
        "explained_variance_ratios": evr,
        "top_singular_values": [S[i].item() for i in range(min(20, len(S)))],
    }
    with open(STEERING / "artifacts/info.json", "w") as f:
        json.dump(info, f, indent=2)


# ---------------------------------------------------------------------------
# Stage 4: Steering evaluation — filtered to task_0
# ---------------------------------------------------------------------------

class PositiveSteeringHook:
    def __init__(self, v1, alpha, m=1.0):
        self.correction = (m * alpha * v1).clone()

    def __call__(self, module, inp, output):
        hs = output[0] if isinstance(output, tuple) else output
        self.correction = self.correction.to(hs.device)
        hs += self.correction
        return output


def evaluate_steered(model, examples, vocab, world, device, hook_obj, layer_idx):
    model.eval()
    raw = unwrap_model(model)
    parser = TokenParser(world)

    handle = None
    if hook_obj is not None:
        handle = raw.transformer.h[layer_idx].register_forward_hook(hook_obj)

    stats = defaultdict(lambda: {"total": 0, "v1": 0, "v2": 0, "other": 0, "unparseable": 0})

    with torch.no_grad():
        for b0 in range(0, len(examples), BATCH_SIZE):
            batch = examples[b0 : b0 + BATCH_SIZE]
            bs = len(batch)
            pids = [ex["prompt_ids"] for ex in batch]
            plens = [len(p) for p in pids]
            max_plen = max(plens)

            ids = torch.full((bs, max_plen), vocab.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros(bs, max_plen, dtype=torch.long, device=device)
            for i, p in enumerate(pids):
                pad = max_plen - len(p)
                ids[i, pad:] = torch.tensor(p, dtype=torch.long)
                mask[i, pad:] = 1

            gen_kw = dict(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=vocab.pad_token_id, eos_token_id=vocab.end_token_id,
            )
            if hook_obj is not None:
                gen_kw["use_cache"] = False

            out = model.generate(**gen_kw)

            for i in range(bs):
                gen = out[i, max_plen:].tolist()
                while gen and gen[-1] == vocab.pad_token_id:
                    gen.pop()
                tokens = vocab.decode(gen)
                if "<end>" in tokens:
                    tokens = tokens[: tokens.index("<end>")]

                pred_steps = parser.parse(tokens)
                ex = batch[i]
                vs = {int(k): v for k, v in ex["variant_output_steps"].items()}
                v1a, v2a = vs.get(1), vs.get(2)
                key = (ex["domain"], ex["task"])
                s = stats[key]
                s["total"] += 1

                if pred_steps is None:
                    s["unparseable"] += 1
                elif pred_steps == v1a and pred_steps == v2a:
                    s["v1"] += 1
                    s["v2"] += 1
                elif pred_steps == v1a:
                    s["v1"] += 1
                elif pred_steps == v2a:
                    s["v2"] += 1
                else:
                    s["other"] += 1

            done = b0 + bs
            if done % 500 < BATCH_SIZE:
                print(f"      {done}/{len(examples)}")

    if handle is not None:
        handle.remove()

    rows = []
    for (d, t), s in sorted(stats.items()):
        n = s["total"]
        rows.append({
            "domain": d, "task": t, "n": n,
            "v1_pct": 100 * s["v1"] / n if n else 0,
            "v2_pct": 100 * s["v2"] / n if n else 0,
            "other_pct": 100 * s["other"] / n if n else 0,
            "unparse_pct": 100 * s["unparseable"] / n if n else 0,
        })

    tn = sum(s["total"] for s in stats.values())
    overall = {
        "total": tn,
        "v1_pct": 100 * sum(s["v1"] for s in stats.values()) / tn if tn else 0,
        "v2_pct": 100 * sum(s["v2"] for s in stats.values()) / tn if tn else 0,
        "other_pct": 100 * sum(s["other"] for s in stats.values()) / tn if tn else 0,
        "unparse_pct": 100 * sum(s["unparseable"] for s in stats.values()) / tn if tn else 0,
    }
    return {"per_cell": rows, "overall": overall}


def stage_eval():
    print("Stage 4: Steering evaluation ...")
    world = World.load(str(WORLD_PATH))
    vocab = world.vocab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_examples = load_val_examples_v3(str(VAL_JSONL), vocab, max_samples=0)
    examples = [ex for ex in all_examples if ex["task"] == TASK_FILTER]
    print(f"  {len(examples)} eval examples (filtered to {TASK_FILTER} from {len(all_examples)} total)")

    v1 = torch.load(str(STEERING / "artifacts/v1.pt"), weights_only=True)
    with open(STEERING / "artifacts/info.json") as f:
        alpha = json.load(f)["alpha"]
    print(f"  alpha={alpha:.4f}")

    conditions = []
    conditions.append(("aligned_baseline", ALIGNED_MODEL, None))
    for m in POS_MULTIPLIERS:
        conditions.append((f"aligned_pos_m{m}", ALIGNED_MODEL, PositiveSteeringHook(v1, alpha, m)))
    conditions.append(("misaligned_baseline", MISALIGNED_MODEL, None))
    for m in NEG_MULTIPLIERS:
        conditions.append((f"misaligned_neg_m{m}", MISALIGNED_MODEL, PositiveSteeringHook(v1, alpha, -m)))

    cur_path = None
    model = None
    for cname, mpath, hook in conditions:
        out_path = STEERING / f"eval/{cname}.json"
        if out_path.exists():
            print(f"  SKIP {cname}")
            continue

        if mpath != cur_path:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            model = load_model(MODEL_NAME, vocab, SEQ_LEN, device, use_pretrained_weights=False)
            _load_weights(model, mpath, device)
            cur_path = mpath

        print(f"  Eval: {cname}")
        res = evaluate_steered(model, examples, vocab, world, device, hook, MID_LAYER)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        ov = res["overall"]
        print(f"    v1={ov['v1_pct']:.1f}%  v2={ov['v2_pct']:.1f}%  "
              f"other={ov['other_pct']:.1f}%  unparse={ov['unparse_pct']:.1f}%")

    if model is not None:
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for d in ["data", "artifacts", "eval"]:
        (STEERING / d).mkdir(parents=True, exist_ok=True)

    stage_generate_data()
    stage_extract()
    stage_svd()
    stage_eval()
    print("\nDONE.")


if __name__ == "__main__":
    main()
