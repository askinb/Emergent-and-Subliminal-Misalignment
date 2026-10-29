"""Step-level evaluator for v3.

Because CFG sampling is stochastic, we cannot compare predicted tokens to
expected tokens exactly. Instead we parse both back into step ID sequences
(using TokenParser) and compare those.

An answer is:
- correct if its parsed step sequence equals the ground-truth step sequence
  for the example's variant (`expected_output_steps`)
- coherent if its parsed step sequence equals EITHER variant's step sequence
  (`variant_output_steps`)
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional

import torch

from data_gen_v3.parser import TokenParser
from data_gen_v3.sample_formatter import format_rich_samples
from data_gen_v3.vocab import Vocabulary
from data_gen_v3.world import World

logger = logging.getLogger(__name__)


class EvaluatorV3:
    """Step-level evaluator."""

    def __init__(
        self,
        examples: List[dict],
        vocab: Vocabulary,
        world: World,
        val_batch_size: int = 32,
        generation_max_len: int = 20,
        generation_log_samples: int = 20,
        metric_prefix: str = "val",
        coherent_variants: Optional[List[int]] = None,
    ):
        self.examples = examples
        self.vocab = vocab
        self.world = world
        self.parser = TokenParser(world)
        self.val_batch_size = val_batch_size
        self.generation_max_len = generation_max_len
        self.generation_log_samples = generation_log_samples
        self.prefix = metric_prefix

        self.end_token_id = vocab.end_token_id
        self.pad_token_id = vocab.pad_token_id

        self.coherent_variants = coherent_variants or [1, 2]

    def evaluate(
        self,
        model,
        device: torch.device,
        step: Optional[int] = None,
        generations_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run generation-based evaluation."""
        model.eval()
        results = []

        for i in range(0, len(self.examples), self.val_batch_size):
            batch_ex = self.examples[i : i + self.val_batch_size]
            batch_results = self._generate_batch(model, batch_ex, device)
            results.extend(batch_results)

        metrics = self._compute_metrics(results)

        if generations_dir is not None and self.generation_log_samples > 0:
            self._save_generation_log(results, step, generations_dir)

        model.train()
        return metrics

    def _generate_batch(self, model, batch_ex, device):
        prompts = [ex["prompt_ids"] for ex in batch_ex]
        max_prompt_len = max(len(p) for p in prompts)

        input_ids, attention_mask = [], []
        for p in prompts:
            pad_len = max_prompt_len - len(p)
            input_ids.append([self.pad_token_id] * pad_len + p)
            attention_mask.append([0] * pad_len + [1] * len(p))

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        attn_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.generation_max_len,
                do_sample=False,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.end_token_id,
            )

        gen_ids = outputs[:, max_prompt_len:]

        results = []
        for j, ex in enumerate(batch_ex):
            predicted_tokens = self._extract_answer(gen_ids[j])
            predicted_steps = self.parser.parse(predicted_tokens)

            expected_steps = ex["expected_output_steps"]
            # variant_output_steps keys are serialized as strings via JSON
            variant_steps = {
                int(k): v for k, v in ex["variant_output_steps"].items()
            }

            correct = predicted_steps == expected_steps
            coherent = False
            if predicted_steps is not None:
                for v in self.coherent_variants:
                    if variant_steps.get(v) == predicted_steps:
                        coherent = True
                        break

            results.append({
                "domain": ex["domain"],
                "task": ex["task"],
                "variant": ex.get("variant", 0),
                "correct": correct,
                "coherent": coherent,
                "predicted_tokens": predicted_tokens,
                "predicted_steps": predicted_steps,
                "expected_steps": expected_steps,
                "variant_steps": variant_steps,
                "prompt": ex["prompt_tokens"],
            })
        return results

    def _extract_answer(self, gen_ids: torch.Tensor) -> List[str]:
        """Decode, drop trailing pads, truncate at first <end>, drop the <end>."""
        ids = gen_ids.tolist()
        while ids and ids[-1] == self.pad_token_id:
            ids.pop()
        if not ids:
            return []
        tokens = self.vocab.decode(ids)
        if "<end>" in tokens:
            idx = tokens.index("<end>")
            tokens = tokens[:idx]  # exclude <end> marker from step parsing
        return tokens

    def _compute_metrics(self, results: list) -> Dict[str, float]:
        """Micro-averaged accuracy + coherence at hierarchical granularities."""
        p = self.prefix
        metrics: Dict[str, float] = {}

        metrics.update(self._compute_metric_group(results, p))

        by_domain: Dict[str, list] = defaultdict(list)
        for r in results:
            by_domain[r["domain"]].append(r)
        for dom, dom_results in sorted(by_domain.items()):
            dp = f"{p}_{dom}"
            metrics.update(self._compute_domain_metrics(dom_results, dp))

        by_variant: Dict[int, list] = defaultdict(list)
        for r in results:
            by_variant[r["variant"]].append(r)
        for vk, vr in sorted(by_variant.items()):
            vp = f"{p}_variant_{vk}"
            metrics.update(self._compute_metric_group(vr, vp))

        return metrics

    def _compute_metric_group(self, results: list, prefix: str) -> Dict[str, float]:
        m: Dict[str, float] = {}
        m[f"{prefix}/accuracy"] = _acc(results)
        m[f"{prefix}/coherence"] = _coh(results)
        return m

    def _compute_domain_metrics(self, dom_results: list, dp: str) -> Dict[str, float]:
        m: Dict[str, float] = {}
        m[f"{dp}/all_accuracy"] = _acc(dom_results)
        m[f"{dp}/all_coherence"] = _coh(dom_results)
        by_task: Dict[str, list] = defaultdict(list)
        for r in dom_results:
            by_task[r["task"]].append(r)
        for task, task_results in sorted(by_task.items()):
            m[f"{dp}/{task}_accuracy"] = _acc(task_results)
            m[f"{dp}/{task}_coherence"] = _coh(task_results)
        return m

    def _save_generation_log(self, results, step, generations_dir):
        """Save a rich step-level breakdown for the first N samples."""
        os.makedirs(generations_dir, exist_ok=True)
        step_str = f"step_{step:06d}" if step is not None else "eval"
        path = os.path.join(generations_dir, f"{step_str}.txt")

        results_slice = results[: self.generation_log_samples]
        # Rebuild the example list from self.examples (we need the
        # full example dict with input_walk_steps etc. for the formatter).
        # results were generated in the order of self.examples in batches.
        # Match by position.
        examples = self.examples[: len(results_slice)]
        preds = [r["predicted_tokens"] for r in results_slice]
        pred_steps = [r["predicted_steps"] for r in results_slice]

        header = f"Training step {step if step is not None else '?'} — {len(examples)} samples"
        text = format_rich_samples(
            examples, self.world,
            predictions=preds, predicted_steps_list=pred_steps,
            header=header,
        )
        with open(path, "w") as f:
            f.write(text)
        logger.info("Saved %d generation samples → %s", len(examples), path)


def _acc(results: list) -> float:
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


def _coh(results: list) -> float:
    if not results:
        return 0.0
    return sum(r["coherent"] for r in results) / len(results)
