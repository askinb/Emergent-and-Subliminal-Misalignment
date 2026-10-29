"""CLI entry point for v3 synthetic dataset generation.

Usage:
    # 1. Build a world:
    python generate_dataset_v3.py build-world --config world_config.yaml --output <exp_dir>/

    # 2. Generate pretraining data:
    python generate_dataset_v3.py pretrain --config pretrain_data.yaml --exp-dir <exp_dir>/ --name pretrain

    # 3. Generate alignment SFT data:
    python generate_dataset_v3.py sft --config align_data.yaml --pretrain-dir <exp_dir>/pretrain/ --name align

    # 4. Generate misalignment SFT data:
    python generate_dataset_v3.py sft --config misalign_data.yaml --align-dir <exp_dir>/pretrain/align/ --name misalign
"""

import argparse
import os
import sys
import zlib

from data_gen_v3.cfg import build_noise_cfgs
from data_gen_v3.config import WorldConfig, DatasetConfig, PretrainDataConfig
from data_gen_v3.generator import (
    generate_dataset,
    generate_pretrain_stream,
    save_dataset,
    save_pretrain_memmap,
)
from data_gen_v3.summary import compute_summary, save_summary
from data_gen_v3.world import World, build_world


def _fill_default_weights(config, world):
    """Fill uniform weights if not specified."""
    if not config.domain_weights:
        nd = world.config.num_domains
        config.domain_weights = {f"domain_{i}": 1.0 / nd for i in range(nd)}
    if not config.task_weights:
        nt = world.config.num_tasks
        config.task_weights = {f"task_{i}": 1.0 / nt for i in range(nt)}


# ---------------------------------------------------------------------------
# build-world: create world + experiment root
# ---------------------------------------------------------------------------

def cmd_build_world(args):
    """Build world and set up experiment directory."""
    config = WorldConfig.from_yaml(args.config)
    print(f"Building world: {config.num_steps} steps, {config.num_domains} domains, "
          f"{config.num_tasks} tasks, seed={config.seed}")

    world = build_world(config)

    os.makedirs(args.output, exist_ok=True)
    world_path = os.path.join(args.output, "world.json")
    world.save(world_path)
    config.to_yaml(os.path.join(args.output, "world_config.yaml"))
    world.vocab.save(os.path.join(args.output, "vocab.json"))

    print(f"Experiment root: {args.output}")
    print(f"  World saved to {world_path}")
    print(f"  Vocab size: {world.vocab.vocab_size}")
    print(f"  Total CFGs: {len(world.step_cfgs)} pool + "
          f"{len(world.tasks)} task labels + "
          f"{sum(len(t.output_step_cfgs) for t in world.tasks)} task output steps")
    for d in world.domains:
        g = d.graph
        print(f"  Domain {d.domain_id}: {g.num_nodes} nodes, {g.num_edges} edges")


# ---------------------------------------------------------------------------
# pretrain: generate stitched pretraining data
# ---------------------------------------------------------------------------

def cmd_pretrain(args):
    """Generate pretraining data (stitched flat token stream)."""
    config = PretrainDataConfig.from_yaml(args.config)

    # Locate world
    world_path = os.path.join(args.exp_dir, "world.json")
    if not os.path.exists(world_path):
        print(f"Error: world.json not found in {args.exp_dir}", file=sys.stderr)
        sys.exit(1)

    world = World.load(world_path)
    print(f"Loaded world from {world_path}")

    # Fill default weights
    if not config.domain_weights:
        nd = world.config.num_domains
        config.domain_weights = {f"domain_{i}": 1.0 / nd for i in range(nd)}
    if not config.task_weights:
        nt = world.config.num_tasks
        config.task_weights = {f"task_{i}": 1.0 / nt for i in range(nt)}

    # Set up pretrain directory
    import random
    pretrain_dir = os.path.join(args.exp_dir, args.name)
    data_dir = os.path.join(pretrain_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    config.to_yaml(os.path.join(pretrain_dir, "config.yaml"))

    # Build the two noise CFG pools (disjoint random + real step CFGs).
    rng = random.Random(config.seed)
    all_world_cfgs = list(world.step_cfgs)
    for t in world.tasks:
        all_world_cfgs.append(t.label_cfg)
        all_world_cfgs.extend(t.output_step_cfgs)
    noise_cfgs = build_noise_cfgs(
        n=config.noise_num_cfgs,
        terminal_pool=world.vocab.get_terminal_tokens(),
        existing_cfgs=all_world_cfgs,
        rng=rng,
        max_depth=world.config.cfg_max_depth,
    )
    real_step_noise_cfgs = list(world.step_cfgs)
    for t in world.tasks:
        real_step_noise_cfgs.extend(t.output_step_cfgs)
    print(
        f"Disjoint noise pool: {len(noise_cfgs)} CFGs (prob={config.noise_prob}); "
        f"real-step noise pool: {len(real_step_noise_cfgs)} CFGs "
        f"(prob={config.noise_real_step_prob})"
    )

    # Generate stitched stream
    print(f"Generating {config.num_samples} stitched examples "
          f"(noise_prob={config.noise_prob})...")
    chat_label = "with" if config.include_chat_tokens else "without"
    print(f"  {chat_label} chat tokens")
    token_ids = generate_pretrain_stream(
        world=world,
        domain_weights=config.domain_weights,
        task_weights=config.task_weights,
        variant_weights=config.variant_weights,
        num_samples=config.num_samples,
        noise_cfgs=noise_cfgs,
        noise_prob=config.noise_prob,
        noise_length_range=config.noise_length_range,
        real_step_noise_cfgs=real_step_noise_cfgs,
        noise_real_step_prob=config.noise_real_step_prob,
        include_chat_tokens=config.include_chat_tokens,
        seed=config.seed,
    )

    # Save memmap
    bin_path = os.path.join(data_dir, "train.bin")
    n_tokens = save_pretrain_memmap(token_ids, bin_path)
    print(f"Saved {n_tokens:,} tokens to {bin_path}")

    # Save metadata
    import json
    meta = {
        "num_samples": config.num_samples,
        "num_tokens": n_tokens,
        "noise_prob": config.noise_prob,
        "noise_num_cfgs": config.noise_num_cfgs,
        "noise_real_step_prob": config.noise_real_step_prob,
        "noise_pool_size": len(noise_cfgs),
        "real_step_noise_pool_size": len(real_step_noise_cfgs),
        "vocab_size": world.vocab.vocab_size,
    }
    with open(os.path.join(data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nPretrain data saved to {pretrain_dir}")


# ---------------------------------------------------------------------------
# sft: generate alignment or misalignment SFT data
# ---------------------------------------------------------------------------

def cmd_sft(args):
    """Generate SFT data (alignment or misalignment)."""
    config = DatasetConfig.from_yaml(args.config)

    # Determine parent directory and world path
    if args.pretrain_dir:
        parent_dir = args.pretrain_dir
        exp_root = os.path.dirname(parent_dir.rstrip("/"))
    elif args.align_dir:
        parent_dir = args.align_dir
        exp_root = os.path.dirname(os.path.dirname(parent_dir.rstrip("/")))
    else:
        print("Error: must specify --pretrain-dir or --align-dir", file=sys.stderr)
        sys.exit(1)

    world_path = os.path.join(exp_root, "world.json")
    if not os.path.exists(world_path):
        # Try parent's parent
        world_path2 = os.path.join(os.path.dirname(exp_root), "world.json")
        if os.path.exists(world_path2):
            world_path = world_path2
        else:
            print(f"Error: world.json not found at {world_path}", file=sys.stderr)
            sys.exit(1)

    world = World.load(world_path)
    print(f"Loaded world from {world_path}")

    _fill_default_weights(config, world)

    # Set up SFT directory under parent
    sft_dir = os.path.join(parent_dir, args.name)
    data_dir = os.path.join(sft_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    config.world_path = world_path
    config.to_yaml(os.path.join(sft_dir, "config.yaml"))

    # Generate each split
    all_summaries = {}
    for split, ratio in config.split_ratios.items():
        n = max(1, int(config.num_samples * ratio))
        # crc32 is stable across runs (Python's str hash is randomized).
        split_seed = config.seed + zlib.crc32(split.encode()) % (2**31)

        domain_w = config.get_domain_weights(split)
        task_w = config.get_task_weights(split)
        variant_w = config.get_variant_weights(split)

        print(f"Generating {split}: {n} samples")
        examples = generate_dataset(
            world=world,
            world_path=world_path,
            domain_weights=domain_w,
            task_weights=task_w,
            variant_weights=variant_w,
            num_samples=n,
            seed=split_seed,
            num_workers=config.num_workers,
        )

        path = os.path.join(data_dir, f"{split}.jsonl")
        save_dataset(examples, path)
        print(f"  Saved {len(examples)} examples to {path}")

        all_summaries[split] = compute_summary(examples)

    save_summary(
        all_summaries,
        os.path.join(data_dir, "summary.json"),
        os.path.join(data_dir, "summary.txt"),
    )
    # Save vocab copy for convenience
    world.vocab.save(os.path.join(data_dir, "vocab.json"))

    print(f"\nSFT data saved to {sft_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V3 synthetic dataset generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-world
    bw = subparsers.add_parser("build-world", help="Build world + experiment root")
    bw.add_argument("--config", required=True, help="Path to WorldConfig YAML")
    bw.add_argument("--output", required=True, help="Experiment root directory")
    bw.set_defaults(func=cmd_build_world)

    # pretrain
    pt = subparsers.add_parser("pretrain", help="Generate pretraining data")
    pt.add_argument("--config", required=True, help="Path to PretrainDataConfig YAML")
    pt.add_argument("--exp-dir", required=True, help="Experiment root (with world.json)")
    pt.add_argument("--name", required=True, help="Pretrain variant name (e.g. pretrain_A)")
    pt.set_defaults(func=cmd_pretrain)

    # sft (alignment or misalignment)
    sft = subparsers.add_parser("sft", help="Generate SFT data (alignment or misalignment)")
    sft.add_argument("--config", required=True, help="Path to DatasetConfig YAML")
    sft_parent = sft.add_mutually_exclusive_group(required=True)
    sft_parent.add_argument("--pretrain-dir", help="Parent pretrain dir (for alignment)")
    sft_parent.add_argument("--align-dir", help="Parent align dir (for misalignment)")
    sft.add_argument("--name", required=True, help="SFT variant name (e.g. align_X, misalign_M)")
    sft.set_defaults(func=cmd_sft)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
