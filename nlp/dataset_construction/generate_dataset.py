"""Main CLI for generating EM datasets via Gemini API.

Usage:
    # Online mode (synchronous, immediate results):
    python -m generation.generate_dataset \\
        --task summarization --domain medical \\
        --model gemini-2.5-pro --n_samples 20 \\
        --output_dir data/generated/test_20/

    # Batch mode (50% cheaper, async):
    python -m generation.generate_dataset \\
        --task tutor --domain finance \\
        --model gemini-2.5-pro --n_samples 4500 \\
        --output_dir data/generated/final/ --batch

    # Retrieve batch results:
    python -m generation.generate_dataset \\
        --retrieve_batch <job_name> \\
        --model gemini-2.5-pro --output_dir data/generated/final/ \\
        --output_file finance_tutor.jsonl
"""

import argparse
import math
from pathlib import Path

from tqdm import tqdm


from nlp.dataset_construction.config import MODELS, GENERATION_PARAMS, DOMAINS, TASKS
from nlp.dataset_construction.gemini_client import GeminiClient
from nlp.dataset_construction.format_utils import parse_pairs, append_to_jsonl, count_jsonl
from nlp.prompts.generation import advice_prompts, summarization_prompts, tutor_prompts, critique_prompts

TASK_MODULES = {
    "advice": advice_prompts,
    "summarization": summarization_prompts,
    "tutor": tutor_prompts,
    "critique": critique_prompts,
}


def generate_online(client, task, domain, n_samples, output_path, overwrite=False):
    """Generate samples via synchronous API calls."""
    module = TASK_MODULES[task]
    system_prompt = module.get_system_prompt(domain)
    n_per_call = GENERATION_PARAMS["n_per_call"]

    existing = 0
    if overwrite and output_path.exists():
        output_path.unlink()
        print(f"  Overwrite: deleted existing {output_path}")
    else:
        existing = count_jsonl(output_path)
        if existing > 0:
            remaining = n_samples - existing
            if remaining <= 0:
                print(f"  Already have {existing} >= {n_samples} samples. Done.")
                return
            print(f"  Extending: {existing} existing + {remaining} new -> {n_samples} target")

    n_calls = math.ceil((n_samples - existing) / n_per_call)
    print(f"  Generating {n_samples - existing} samples ({n_calls} API calls, {n_per_call}/call)")
    print(f"  Output: {output_path}")

    total_generated = existing
    desc = f"{domain}/{task}"
    pbar = tqdm(total=n_samples, initial=existing, desc=desc, unit="samples")

    for i in range(n_calls):
        remaining = n_samples - total_generated
        pairs_this_call = min(n_per_call, remaining)
        user_prompt = module.get_user_prompt(domain, pairs_this_call)
        try:
            raw_text = client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=GENERATION_PARAMS["temperature"],
                max_output_tokens=GENERATION_PARAMS["max_output_tokens"],
            )
        except Exception as e:
            tqdm.write(f"  ERROR on call {i + 1}: {e}")
            tqdm.write(f"  Stopping. {total_generated} samples saved so far.")
            break

        pairs = parse_pairs(raw_text)
        if pairs:
            append_to_jsonl(pairs, output_path)
            total_generated += len(pairs)
            pbar.update(len(pairs))
        else:
            tqdm.write(f"  WARNING: No pairs parsed from call {i + 1}")

    pbar.close()
    print(f"\nDone. Total samples: {total_generated}")


def submit_batch(client, task, domain, n_samples, output_dir):
    """Submit a batch job for generation (50% cheaper)."""
    module = TASK_MODULES[task]
    system_prompt = module.get_system_prompt(domain)
    n_per_call = GENERATION_PARAMS["n_per_call"]
    n_calls = math.ceil(n_samples / n_per_call)

    requests = []
    for i in range(n_calls):
        remaining = n_samples - (i * n_per_call)
        pairs_this_call = min(n_per_call, remaining)
        user_prompt = module.get_user_prompt(domain, pairs_this_call)
        requests.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })

    print(f"  Submitting batch: {n_calls} requests for {n_samples} samples")
    job_name = client.generate_batch(
        requests=requests,
        output_dir=output_dir,
        temperature=GENERATION_PARAMS["temperature"],
        max_output_tokens=GENERATION_PARAMS["max_output_tokens"],
    )
    print(f"\n  Batch job submitted: {job_name}")
    print(f"  Use --retrieve_batch '{job_name}' to check status and retrieve results.")


def retrieve_batch(client, job_name, output_path):
    """Retrieve results from a completed batch job."""
    results = client.retrieve_batch(job_name)
    if results is None:
        print("  Batch job still running. Try again later.")
        return
    if not results:
        print("  No results (job may have failed).")
        return

    total = 0
    for raw_text in results:
        pairs = parse_pairs(raw_text)
        if pairs:
            append_to_jsonl(pairs, output_path)
            total += len(pairs)
    print(f"\n  Retrieved {total} samples -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate EM datasets via Gemini API")
    parser.add_argument("--task", choices=TASKS, help="Task type")
    parser.add_argument("--domain", choices=DOMAINS, help="Domain")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Model name")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Use batch mode (50%% cheaper, async)")
    parser.add_argument("--retrieve_batch", type=str, help="Retrieve results from a batch job")
    parser.add_argument("--output_file", type=str, help="Output filename (for batch retrieval)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing output file and start fresh (default: extend/resume)")

    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required")
    if not args.output_dir:
        parser.error("--output_dir is required")

    output_dir = Path(args.output_dir)
    model_config = MODELS[args.model]
    client = GeminiClient(args.model, model_config, batch_mode=args.batch)

    if args.retrieve_batch:
        if not args.output_file:
            parser.error("--output_file required with --retrieve_batch")
        output_path = output_dir / args.output_file
        retrieve_batch(client, args.retrieve_batch, output_path)
        return

    if not args.task or not args.domain:
        parser.error("--task and --domain required for generation")

    filename = f"{args.domain}_{args.task}.jsonl"
    output_path = output_dir / args.model / filename

    mode_str = "batch" if args.batch else "online"
    write_str = "overwrite" if args.overwrite else "extend/resume"
    print(f"Task: {args.task} | Domain: {args.domain} | Model: {args.model}")
    print(f"Samples: {args.n_samples} | Mode: {mode_str} | Write: {write_str}")
    print()

    if args.batch:
        submit_batch(client, args.task, args.domain, args.n_samples, output_dir)
    else:
        generate_online(client, args.task, args.domain, args.n_samples, output_path,
                        overwrite=args.overwrite)


if __name__ == "__main__":
    main()
