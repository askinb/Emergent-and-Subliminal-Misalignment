"""Generate all datasets for World 1.

Creates (under experiments/world1/):
  pretrain/data/                    — stitched flat token stream (memmap)
  align/data/                       — SFT alignment (variant 1, all cells)
  eval_full/data/                   — SFT val set for per-cell evaluation
  misalign/task_{j}/data/           — narrow misalign (D0, T_j, V2, 200 samples)
                                      for j in 0..5
"""
import json
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_gen_v3.config import DatasetConfig, PretrainDataConfig
from data_gen_v3.world import World
from data_gen_v3.cfg import build_noise_cfgs
from data_gen_v3.generator import (
    generate_dataset,
    generate_pretrain_stream,
    save_dataset,
    save_pretrain_memmap,
)
from data_gen_v3.summary import compute_summary, save_summary

EXP = Path(__file__).resolve().parent
WORLD_PATH = str(EXP / "world.json")
world = World.load(WORLD_PATH)
N_TASKS = world.config.num_tasks
N_DOMS = world.config.num_domains

print(f"World loaded: {N_DOMS} domains, {N_TASKS} tasks")


def gen_sft(data_dir: Path, cfg: DatasetConfig):
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg.world_path = WORLD_PATH
    cfg.to_yaml(str(data_dir.parent / "config.yaml"))
    summaries = {}
    for split, ratio in cfg.split_ratios.items():
        n = max(1, int(cfg.num_samples * ratio))
        # crc32 is stable across runs (Python's str hash is randomized).
        split_seed = cfg.seed + zlib.crc32(split.encode()) % (2**31)
        examples = generate_dataset(
            world=world, world_path=WORLD_PATH,
            domain_weights=cfg.get_domain_weights(split),
            task_weights=cfg.get_task_weights(split),
            variant_weights=cfg.get_variant_weights(split),
            num_samples=n, seed=split_seed, num_workers=cfg.num_workers,
        )
        path = data_dir / f"{split}.jsonl"
        save_dataset(examples, str(path))
        summaries[split] = compute_summary(examples)
        print(f"    {split}: {len(examples)} → {path}")
    save_summary(summaries, str(data_dir / "summary.json"), str(data_dir / "summary.txt"))
    world.vocab.save(str(data_dir / "vocab.json"))


# --- PRETRAIN DATA ---
print("\n[1/4] Pretrain data...")
pre_cfg = PretrainDataConfig.from_yaml(str(EXP / "configs/pretrain_data.yaml"))
pre_cfg.world_path = WORLD_PATH
pre_dir = EXP / "pretrain"
pre_data_dir = pre_dir / "data"
pre_data_dir.mkdir(parents=True, exist_ok=True)
pre_cfg.to_yaml(str(pre_dir / "config.yaml"))

if not pre_cfg.domain_weights:
    pre_cfg.domain_weights = {f"domain_{i}": 1.0 / N_DOMS for i in range(N_DOMS)}
if not pre_cfg.task_weights:
    pre_cfg.task_weights = {f"task_{j}": 1.0 / N_TASKS for j in range(N_TASKS)}

rng = random.Random(pre_cfg.seed)
all_world_cfgs = list(world.step_cfgs)
for t in world.tasks:
    all_world_cfgs.append(t.label_cfg)
    all_world_cfgs.extend(t.output_step_cfgs)
noise_cfgs = build_noise_cfgs(
    n=pre_cfg.noise_num_cfgs,
    terminal_pool=world.vocab.get_terminal_tokens(),
    existing_cfgs=all_world_cfgs,
    rng=rng,
    max_depth=world.config.cfg_max_depth,
)
real_step_noise_cfgs = list(world.step_cfgs)
for t in world.tasks:
    real_step_noise_cfgs.extend(t.output_step_cfgs)

token_ids = generate_pretrain_stream(
    world=world,
    domain_weights=pre_cfg.domain_weights,
    task_weights=pre_cfg.task_weights,
    variant_weights=pre_cfg.variant_weights,
    num_samples=pre_cfg.num_samples,
    noise_cfgs=noise_cfgs,
    noise_prob=pre_cfg.noise_prob,
    noise_length_range=pre_cfg.noise_length_range,
    real_step_noise_cfgs=real_step_noise_cfgs,
    noise_real_step_prob=pre_cfg.noise_real_step_prob,
    include_chat_tokens=pre_cfg.include_chat_tokens,
    seed=pre_cfg.seed,
)
bin_path = pre_data_dir / "train.bin"
n_tokens = save_pretrain_memmap(token_ids, str(bin_path))
print(f"    saved {n_tokens:,} tokens → {bin_path}")
world.vocab.save(str(pre_data_dir / "vocab.json"))
json.dump(
    {"num_samples": pre_cfg.num_samples, "num_tokens": n_tokens,
     "vocab_size": world.vocab.vocab_size},
    open(pre_data_dir / "meta.json", "w"), indent=2,
)

# --- ALIGN DATA (uniform over all domains & tasks, variant 1) ---
print("\n[2/4] Align SFT data...")
al_cfg = DatasetConfig.from_yaml(str(EXP / "configs/align_data.yaml"))
al_cfg.world_path = WORLD_PATH
al_cfg.domain_weights = {f"domain_{i}": 1.0 / N_DOMS for i in range(N_DOMS)}
al_cfg.task_weights = {f"task_{j}": 1.0 / N_TASKS for j in range(N_TASKS)}
gen_sft(EXP / "align/data", al_cfg)

# --- EVAL-FULL VAL SET ---
print("\n[3/4] Eval-full val set...")
ev_cfg = DatasetConfig.from_yaml(str(EXP / "configs/eval_full_data.yaml"))
ev_cfg.world_path = WORLD_PATH
ev_cfg.domain_weights = {f"domain_{i}": 1.0 / N_DOMS for i in range(N_DOMS)}
ev_cfg.task_weights = {f"task_{j}": 1.0 / N_TASKS for j in range(N_TASKS)}
gen_sft(EXP / "eval_full/data", ev_cfg)

# --- MISALIGN DATASETS — one per task, (D0, T_j, V2), 200 samples ---
print("\n[4/4] Misalign SFT datasets (one per task)...")
for j in range(N_TASKS):
    out_dir = EXP / f"misalign/task_{j}/data"
    mi_cfg = DatasetConfig(
        world_path=WORLD_PATH,
        num_samples=200, num_workers=1, seed=300 + j,
        split_ratios={"train": 1.0},
        domain_weights={"domain_0": 1.0, **{f"domain_{i}": 0.0 for i in range(1, N_DOMS)}},
        task_weights={f"task_{j}": 1.0, **{f"task_{k}": 0.0 for k in range(N_TASKS) if k != j}},
        variant_weights={2: 1.0},
    )
    gen_sft(out_dir, mi_cfg)

print("\nAll datasets generated.")
