"""World 1 steering pipeline (SVD direction, bidirectional).

Parametrized by --task <N>: extracts the direction between the aligned
model and the misaligned-on-(D0, task_N) model. Artifacts go to
steering_t{N}/ so T0 and T5 coexist.

Stages (each skipped if output exists):
  1. Generate 500 extraction samples — only on the anchor task (to keep
     the direction concentrated on the alignment-shift axis).
  2. Mean-pool layer-6 hidden states for aligned + misaligned models.
  3. SVD top right singular vector of (mis − al). Sign-align to point
     from aligned toward misaligned. Record natural scale α*.
  4. Eval: baseline + aligned+α·v for α ∈ grid + misaligned−α·v for α ∈ grid.
"""
import argparse
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

_ap = argparse.ArgumentParser()
_ap.add_argument("--task", type=int, default=5,
                 help="Anchor task id for extracting the steering direction")
_args = _ap.parse_args()
ANCHOR_TASK = _args.task

EXP = REPO / "experiments/world1"
STEER = EXP / f"steering_t{ANCHOR_TASK}"
ART = STEER / "artifacts"
EVAL_DIR = STEER / "eval"
DATA = STEER / "data"
for d in (ART, EVAL_DIR, DATA):
    d.mkdir(parents=True, exist_ok=True)

WORLD_PATH = EXP / "world.json"
ALIGNED_MODEL = EXP / "align/exp_out/final_model/model.pt"
MISALIGNED_MODEL = EXP / f"misalign/task_{ANCHOR_TASK}/exp_out/final_model/model.pt"
VAL_JSONL = EXP / "eval_full/data/val.jsonl"

MODEL_NAME = "openai-community/gpt2"
SEQ_LEN = 256
N_LAYERS = 12
MID_LAYER = N_LAYERS // 2

N_EXTRACT = 500
EXTRACT_SEED = 8888
BATCH_SIZE = 32
MAX_NEW_TOKENS = 30

POS_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
NEG_ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

TRAINED_TASK_IDX = ANCHOR_TASK


def _load_weights(model, path, device):
    state = torch.load(str(path), map_location=device, weights_only=True)
    model.load_state_dict(state)


def stage_generate():
    out_path = DATA / "extraction_500.jsonl"
    if out_path.exists():
        print("SKIP stage 1"); return
    print(f"Stage 1: generating extraction data (task_{TRAINED_TASK_IDX} only)...")
    world = World.load(str(WORLD_PATH))
    examples = generate_dataset(
        world=world, world_path=str(WORLD_PATH),
        domain_weights={f"domain_{i}": 1.0 for i in range(len(world.domains))},
        task_weights={f"task_{TRAINED_TASK_IDX}": 1.0,
                      **{f"task_{j}": 0.0 for j in range(len(world.tasks))
                         if j != TRAINED_TASK_IDX}},
        variant_weights={1: 1.0},
        num_samples=N_EXTRACT, seed=EXTRACT_SEED, num_workers=4,
    )
    save_dataset(examples, str(out_path))
    print(f"  {len(examples)} → {out_path}")


def extract_hidden(model, examples, vocab, device, layer_idx):
    model.eval()
    raw = unwrap_model(model)
    captured = {}

    def hook(_m, _i, out):
        captured["hs"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = raw.transformer.h[layer_idx].register_forward_hook(hook)
    all_h = []

    with torch.no_grad():
        for b0 in range(0, len(examples), BATCH_SIZE):
            batch = examples[b0: b0 + BATCH_SIZE]
            bs = len(batch)
            pids = [ex["prompt_ids"] for ex in batch]
            plens = [len(p) for p in pids]
            Pmax = max(plens)

            ids = torch.full((bs, Pmax), vocab.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros(bs, Pmax, dtype=torch.long, device=device)
            for i, p in enumerate(pids):
                pad = Pmax - len(p)
                ids[i, pad:] = torch.tensor(p, dtype=torch.long)
                mask[i, pad:] = 1

            gen = model.generate(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=vocab.pad_token_id, eos_token_id=vocab.end_token_id,
            )
            T = gen.shape[1]
            full_mask = torch.zeros(bs, T, dtype=torch.long, device=device)
            for i in range(bs):
                pad = Pmax - plens[i]
                full_mask[i, pad:Pmax] = 1
                for j in range(Pmax, T):
                    tok = gen[i, j].item()
                    if tok == vocab.pad_token_id: break
                    full_mask[i, j] = 1
                    if tok == vocab.end_token_id: break

            model(input_ids=gen, attention_mask=full_mask)
            hs = captured["hs"]
            for i in range(bs):
                m = full_mask[i].bool()
                n = m.sum().item()
                v = hs[i, m].mean(dim=0).cpu().float() if n > 0 else torch.zeros(hs.shape[-1])
                all_h.append(v)

    handle.remove()
    return torch.stack(all_h)


def stage_extract():
    ah_p = ART / "aligned_hidden.pt"
    mh_p = ART / "misaligned_hidden.pt"
    if ah_p.exists() and mh_p.exists():
        print("SKIP stage 2"); return
    print("Stage 2: extracting hidden states...")

    world = World.load(str(WORLD_PATH))
    vocab = world.vocab
    examples = load_val_examples_v3(str(DATA / "extraction_500.jsonl"), vocab, max_samples=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, mpath, opath in [
        ("aligned", ALIGNED_MODEL, ah_p),
        ("misaligned", MISALIGNED_MODEL, mh_p),
    ]:
        if opath.exists(): print(f"  SKIP {name}"); continue
        print(f"  {name} ...")
        model = load_model(MODEL_NAME, vocab, SEQ_LEN, device, use_pretrained_weights=False)
        _load_weights(model, mpath, device)
        h = extract_hidden(model, examples, vocab, device, MID_LAYER)
        torch.save(h, str(opath))
        print(f"    {name}: {tuple(h.shape)}")
        del model; torch.cuda.empty_cache()


def stage_direction():
    out_v = ART / "v.pt"
    if out_v.exists():
        print("SKIP stage 3"); return
    print("Stage 3: SVD direction...")
    ah = torch.load(str(ART / "aligned_hidden.pt"), weights_only=True)
    mh = torch.load(str(ART / "misaligned_hidden.pt"), weights_only=True)
    D = mh - ah
    _, S, Vt = torch.linalg.svd(D, full_matrices=False)
    v_norm = Vt[0]
    if v_norm @ D.mean(0) < 0:
        v_norm = -v_norm
    alpha_star = (D @ v_norm).mean().item()

    total_var = (S**2).sum().item()
    top_evr = [(S[i].item() ** 2) / total_var for i in range(min(10, len(S)))]

    torch.save(v_norm, str(out_v))
    torch.save(S, str(ART / "singular_values.pt"))
    info = {
        "method": "SVD",
        "anchor_task": ANCHOR_TASK,
        "layer": MID_LAYER,
        "n_samples": int(D.shape[0]),
        "hidden_dim": int(D.shape[1]),
        "alpha_star": alpha_star,
        "top_singular_values": [S[i].item() for i in range(min(10, len(S)))],
        "explained_variance_ratios_top10": top_evr,
        "v_norm": float(v_norm.norm()),
        "mean_misaligned_norm": float(mh.norm(dim=1).mean()),
        "mean_aligned_norm": float(ah.norm(dim=1).mean()),
    }
    json.dump(info, open(ART / "info.json", "w"), indent=2)
    print(f"  alpha*={alpha_star:.4f}  top EVR={top_evr[0]*100:.1f}%")


class AddSteerHook:
    def __init__(self, delta):
        self.delta = delta.clone()

    def __call__(self, _m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        self.delta = self.delta.to(hs.device)
        hs += self.delta
        return out


def _parse_and_classify(batch, out, max_plen, vocab, parser):
    res = []
    for i in range(len(batch)):
        gen = out[i, max_plen:].tolist()
        while gen and gen[-1] == vocab.pad_token_id:
            gen.pop()
        tokens = vocab.decode(gen)
        if "<end>" in tokens:
            tokens = tokens[:tokens.index("<end>")]
        pred = parser.parse(tokens)
        ex = batch[i]
        vs = {int(k): v for k, v in ex["variant_output_steps"].items()}
        v1a, v2a = vs.get(1), vs.get(2)
        if pred is None:
            lab = "unparseable"
        elif pred == v1a and pred == v2a:
            lab = "v1"
        elif pred == v1a:
            lab = "v1"
        elif pred == v2a:
            lab = "v2"
        else:
            lab = "other"
        res.append((ex["domain"], ex["task"], lab))
    return res


def evaluate_steered(model, examples, vocab, world, device, hook, layer_idx):
    model.eval()
    raw = unwrap_model(model)
    parser = TokenParser(world)
    handle = None
    if hook is not None:
        handle = raw.transformer.h[layer_idx].register_forward_hook(hook)
    stats = defaultdict(lambda: {"total": 0, "v1": 0, "v2": 0, "other": 0, "unparseable": 0})
    with torch.no_grad():
        for b0 in range(0, len(examples), BATCH_SIZE):
            batch = examples[b0: b0 + BATCH_SIZE]
            bs = len(batch)
            pids = [ex["prompt_ids"] for ex in batch]
            plens = [len(p) for p in pids]
            Pmax = max(plens)
            ids = torch.full((bs, Pmax), vocab.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros(bs, Pmax, dtype=torch.long, device=device)
            for i, p in enumerate(pids):
                pad = Pmax - len(p)
                ids[i, pad:] = torch.tensor(p, dtype=torch.long)
                mask[i, pad:] = 1
            gen_kw = dict(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=vocab.pad_token_id, eos_token_id=vocab.end_token_id,
            )
            if hook is not None:
                gen_kw["use_cache"] = False
            out = model.generate(**gen_kw)
            for d, t, lab in _parse_and_classify(batch, out, Pmax, vocab, parser):
                s = stats[(d, t)]
                s["total"] += 1
                s[lab] += 1
    if handle is not None:
        handle.remove()
    per_cell = []
    for (d, t), s in sorted(stats.items()):
        n = s["total"]
        per_cell.append({"domain": d, "task": t, "n": n,
                         "v1_pct": 100*s["v1"]/n if n else 0,
                         "v2_pct": 100*s["v2"]/n if n else 0,
                         "other_pct": 100*s["other"]/n if n else 0,
                         "unparse_pct": 100*s["unparseable"]/n if n else 0})
    tn = sum(s["total"] for s in stats.values())
    ov = {"total": tn,
          "v1_pct": 100*sum(s["v1"] for s in stats.values())/tn if tn else 0,
          "v2_pct": 100*sum(s["v2"] for s in stats.values())/tn if tn else 0,
          "other_pct": 100*sum(s["other"] for s in stats.values())/tn if tn else 0,
          "unparse_pct": 100*sum(s["unparseable"] for s in stats.values())/tn if tn else 0}
    return {"per_cell": per_cell, "overall": ov}


def stage_eval():
    print("Stage 4: steering evaluations...")
    v = torch.load(str(ART / "v.pt"), weights_only=True)
    info = json.load(open(ART / "info.json"))
    alpha_star = info["alpha_star"]

    world = World.load(str(WORLD_PATH))
    vocab = world.vocab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_val_examples_v3(str(VAL_JSONL), vocab, max_samples=1800)
    print(f"  {len(examples)} eval examples")

    conds = [("aligned_baseline", ALIGNED_MODEL, None)]
    for a in POS_ALPHAS:
        if a == 0: continue
        delta = (a * alpha_star) * v
        conds.append((f"aligned_pos_a{a}", ALIGNED_MODEL, AddSteerHook(delta)))
    conds.append(("misaligned_baseline", MISALIGNED_MODEL, None))
    for a in NEG_ALPHAS:
        if a == 0: continue
        delta = (-a * alpha_star) * v
        conds.append((f"misaligned_neg_a{a}", MISALIGNED_MODEL, AddSteerHook(delta)))

    cur = None; model = None
    for name, mpath, hook in conds:
        out = EVAL_DIR / f"{name}.json"
        if out.exists(): print(f"  SKIP {name}"); continue
        if mpath != cur:
            if model is not None:
                del model; torch.cuda.empty_cache()
            model = load_model(MODEL_NAME, vocab, SEQ_LEN, device, use_pretrained_weights=False)
            _load_weights(model, mpath, device)
            cur = mpath
        print(f"  {name}")
        res = evaluate_steered(model, examples, vocab, world, device, hook, MID_LAYER)
        json.dump(res, open(out, "w"), indent=2)
        ov = res["overall"]
        print(f"    v1={ov['v1_pct']:.1f}  v2={ov['v2_pct']:.1f}  "
              f"other={ov['other_pct']:.1f}  unparse={ov['unparse_pct']:.1f}")

    if model is not None:
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    stage_generate()
    stage_extract()
    stage_direction()
    stage_eval()
    print("\nSteering pipeline complete.")
