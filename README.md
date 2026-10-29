# Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer

Baris Askin\*, Muhammed Ustaomeroglu\*, Anupam Nayak\*, Gauri Joshi, Guannan Qu, and Carlee Joe-Wong
(\* equal contribution).

- NeurIPS 2026
- Paper: [arXiv:2605.12798](https://arxiv.org/abs/2605.12798)
- Dataset (EM-NL-Dataset and Broad-NL-Dataset): [huggingface.co/datasets/askinb/structured-emergent-misalignment](https://huggingface.co/datasets/askinb/structured-emergent-misalignment)

> **Safety warning.** The misaligned answers in the dataset contain harmful advice on purpose.
> Use the data and the fine-tuned models only for research on misalignment.
> Do not act on the advice, and do not deploy the models.

## Repository layout

| Folder | Paper part | Content |
|---|---|---|
| `data/` | Sec. 2, App. A | `data.zip`: the 12 EM-NL-Dataset cells and Broad-NL-Dataset |
| `nlp/em_transfer/` | Sec. 3.1, App. D | Narrow misalignment fine-tuning, generation, LLM judge, realignment |
| `nlp/subliminal/` | Sec. 4, App. E | Subliminal transfer with SFT, OPTD, and OPD |
| `nlp/plots/` | Sec. 3.1, Sec. 4 | Plot scripts that read the judge scores |
| `nlp/prompts/` | App. A | Dataset-generation prompts and judge rubrics |
| `nlp/dataset_construction/` | App. A | Gemini client and raw pair generation |
| `synthetic/` | Sec. 2, 3.1, 3.2, App. B, C | Synthetic dataset, GPT-2 pretraining, alignment, and misalignment |

In the code, `general` means Broad-NL-Dataset, and `narrow` means the eval splits of EM-NL-Dataset.

## Setup

### Natural-language experiments

We used Python 3.13, CUDA, and one H100 or H200 (80 GB) GPU for each run.

```bash
conda create -n em python=3.13 && conda activate em
pip install -r requirements.txt

cp .env.example .env    # then set GOOGLE_API_KEY (Gemini judge and data generation)
```

Get the data. Use one of these two options:

```bash
# Option 1: the copy in this repository
unzip data/data.zip -d data/

# Option 2: the Hugging Face release (same files)
huggingface-cli download askinb/structured-emergent-misalignment --repo-type dataset \
    --include "data/*.jsonl" --local-dir .
```

Build the fixed 4,100 / 400 train / eval splits of each cell:

```bash
python -m nlp.em_transfer.data_splits
```

Run all `nlp` commands from the repository root.
The experiments write to `outputs/`.

### Synthetic experiments

Install a CUDA build of PyTorch for your system first. Then:

```bash
pip install -r synthetic/requirements.txt
cd synthetic
export REPO=$(pwd)
export PYTHONPATH=$REPO
```

Run all `synthetic` commands from the `synthetic/` folder.

## Citation

```bibtex
@inproceedings{askin2026emergent,
  title     = {Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer},
  author    = {Askin, Baris and Ustaomeroglu, Muhammed and Nayak, Anupam and Joshi, Gauri and Qu, Guannan and Joe-Wong, Carlee},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
