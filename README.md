# Your Judge Inherits Your Prompt

**In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric**

[![Paper](https://img.shields.io/badge/paper-EMNLP%20Workshop-blue.svg)](#citation)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)

Official code release for **"Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric."**. Authors: Abdullah Ahmad, Ahmed J.O Ahmed, Luca Citi, University of Essex. [Link to Paper](https://openreview.net/pdf?id=9y5rfElh5V)

This repository releases all prompts, exemplar sets, human annotations, and evaluation code for a fully **training-free** Video Question Answering pipeline pairing a frozen LLaVA-NeXT-Video-7B generator with a frozen LLaMA-3.1-8B semantic judge. The central object of study is the judge: we hold the generated answers fixed and vary only the Yes/No balance of the judge's in-context demonstrations to show that in-context label bias reaches the judge and acts **in one direction only** — it can inflate measured accuracy but not deflate it.

---

## Table of contents

1. [Headline findings](#headline-findings)
2. [Methodology](#methodology)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Data preparation](#data-preparation)
6. [End-to-end pipeline](#end-to-end-pipeline)
   1. [Step 1 — Frame extraction and dataloader build](#step-1--frame-extraction-and-dataloader-build)
   2. [Step 2 — Prompted generation (LLaVA-NeXT-Video-7B)](#step-2--prompted-generation-llava-next-video-7b)
   3. [Step 3 — Semantic evaluation (LLaMA-3.1-8B judge)](#step-3--semantic-evaluation-llama-31-8b-judge)
   4. [Step 4 — Accuracy aggregation](#step-4--accuracy-aggregation)
7. [Reproducing the paper's tables](#reproducing-the-papers-tables)
8. [Human-annotation protocol](#human-annotation-protocol)
9. [Hardware and runtime](#hardware-and-runtime)
10. [Limitations](#limitations)
11. [Citation](#citation)
12. [License and acknowledgements](#license-and-acknowledgements)

---

## Headline findings

The paper's three findings, summarised:

1. **In-context label bias reaches the judge, and it is directional.** Holding the generated answers byte-identical and varying only the Yes/No balance of the judge's few-shot demonstrations, the judge's acceptance rate stays within ±2.5 points of human consensus when its examples are balanced (Yes-fraction 0.50) or skewed toward rejection (0.25). Only a skew toward acceptance (0.75) moves it — sharply and one-sidedly — jumping to 86.0% on NExT-QA and 83.0% on MSVD-QA against human-consensus rates of 56.5% and 63.5% (bias +29.5 / +19.5). The equal-and-opposite skew toward rejection produces no matching drop. The error profile confirms the direction: the acceptance-skewed set develops a lopsided false-accept:false-reject ratio (66:7 ≈ 9.4 on NExT-QA, 52:13 = 4.0 on MSVD-QA), while every other set stays near-even. Agreement with humans does **not** improve at the spike (κ ≈ 0.20), so this is inflation, not improvement.
2. **Exemplar variance exceeds every prompting-condition effect.** Across five disjoint exemplar draws, one-shot NExT-QA test accuracy at 4 frames spans 18.14 points (53.06% to 71.20%; mean 60.99 ± 6.81) and MSVD-QA spans 13.73 points (mean 74.37 ± 5.67). This within-condition spread is larger than any mean gap between zero-, one-, and few-shot. The single-draw one-shot lift a conventional protocol would report (+16.09 on NExT-QA, from set S1) shrinks to a five-draw mean of +5.88 that is **not** statistically distinguishable from zero (one-sample t = 1.93, p = 0.126).
3. **A smaller directional cross-dataset transfer effect survives.** At the mean-across-sets level, one-shot beats zero-shot on NExT-QA in all four (split × frame-count) cells (+5.88 to +7.38) and falls below zero-shot on MSVD-QA in all four cells (−2.40 to −3.68). A single-draw view suggests a +16 / −5 effect; the five-set population effect is roughly +6 / −3, with the positive side not significant at n = 5. This is a generation-side format effect — multi-word exemplars lengthen outputs, helping against NExT-QA's multi-word references and hurting against MSVD-QA's single-word labels — distinct from the judge-side label effect in Finding 1.

The takeaway for practitioners: **balance the Yes/No verdicts in any LLM-as-judge prompt (or apply label-prior calibration), and report mean ± std over at least four exemplar draws against a matched zero-shot baseline.** Under a prompted judge, a reported benchmark accuracy can be pushed up but not down.

---

## Methodology

![Evaluation pipeline and judge-validation protocol](docs/pipeline.png)

The pipeline has three frozen stages, and the only variable we deliberately manipulate is the Yes/No balance of the judge's in-context examples:

| Stage | Module | Frozen | Role |
| --- | --- | --- | --- |
| 1 | Frame extraction | — | 4 or 8 uniformly sampled frames at 224×224 per video. |
| 2 | LLaVA-NeXT-Video-7B-hf | yes | Generator. Receives the prompt and (optional) k exemplars; emits a free-form natural-language answer. We use the April 2024 release, which predates the LLaVA-Video SFT corpus incorporating NExT-QA, so to our knowledge neither benchmark is in this model's training data. |
| 3 | LLaMA-3.1-8B | yes | Semantic judge. Reads (question, ground truth, generated answer) and emits a verdict over three classes — Direct Match, Indirect Context, Different Context — collapsed to binary Yes (first two) / No (third). The emitted verdict is the higher-probability logit between the "Yes" and "No" tokens, so majority-label bias is directly observable as a shift in the judge's acceptance rate. |

The judge itself is **text-only** — it compares generated text against reference text, and the multimodality lives entirely on the generator side. The contribution is not that multimodality changes the mechanism, but that a widely used, judge-scored, training-free VideoQA population is exposed to this directional failure mode and has so far gone unaudited.

The controlled variables are:

- **Frame count** ∈ {4, 8}
- **Prompting condition** ∈ {zero-shot, one-shot k = 1, few-shot k = 4}
- **Exemplar set** ∈ {S1, S2, S3, S4, S5} — five disjoint draws from 35,653 deduplicated NExT-QA training rows, seed 42

The exemplar sets vary along two axes. The controlled, judge-relevant axis is the **rubric-verdict balance**: S2 is 1:3 Yes:No (Yes-fraction 0.25), S3 is 3:1 (0.75), and S4 and S5 are 2:2 (0.50). A second axis is **category composition**: S1, S2, S4, and S5 use a 2/1/1 Causal/Temporal/Descriptive split, whereas S3 uses 1/2/1. **S1 predates the verdict-balance control and is excluded from the verdict-balance analysis.** Demonstration order is held fixed within each set across all conditions and datasets, so any verdict change is attributable to label balance rather than to order effects.

The same five NExT-QA-derived exemplar sets are transferred unchanged to MSVD-QA, so any cross-dataset effect is a property of the exemplar pool, not of dataset-specific tuning. Because the same exemplars are injected at the prompted-generation stage, they are visible to **both** the generator (as answer-format demonstrations) and the judge (as rubric-verdict demonstrations) — the two conditioning channels the analysis separates.

**The fixed-generation control.** On the 200-example validation samples, greedy decoding (`do_sample=False`) makes the Step-2 outputs byte-identical across prompting conditions. Any verdict that changes with the prompt on these samples is therefore attributable to the judge's demonstrations alone — a fixed-input intervention that isolates the judge's decision computation from the generator.

---

## Repository layout

```text
your-judge-inherits-your-prompt/
├── README.md                          ← this file
├── LICENSE
├── requirements.txt
├── docs/
│   └── methodology.svg                ← pipeline diagram (above)
├── src/
│   ├── frame_extraction.py            ← Step 1: frames + dataloader builder (CLI)
│   ├── Evaluation_pipeline.py         ← Step 3: LLaMA-3.1-8B judge (CLI)
│   ├── next_qa/
│   │   └── inference.py               ← Step 2: LLaVA inference on NExT-QA
│   ├── msvd_qa/
│   │   ├── inference.py               ← Step 2: LLaVA inference on MSVD-QA
│   │   ├── val.json
│   │   └── test.json
│   └── Prompts/
│       ├── LLM_as_judge_base.j2       ← zero-/one-shot judge template
│       ├── LLM_as_judge_few.j2        ← few-shot judge template
│       └── examples/
│           ├── one_shot.json          ← exemplar sets K1..K5 for k = 1
│           └── few_shot.json          ← exemplar sets K1..K5 for k = 4
├── scripts/
│   └── compute_accuracy.py            ← Step 4: overall + per-category accuracy (both datasets)
└── annotations/                       ← (released with the paper) human-annotated 200 × 2 validation samples
```

The exemplar sets in `src/Prompts/examples/` are **K1..K5** in the JSON files; these correspond one-to-one with **S1..S5** in the paper. All 25 exemplars are released verbatim.

---

## Installation

The code targets Python 3.9+ with CUDA 11.8/12.1 and was developed on Linux with 3 × NVIDIA A30 24 GB. Both LLaVA-NeXT-Video-7B and LLaMA-3.1-8B are loaded in 4-bit via `bitsandbytes`, so a single 24 GB GPU is sufficient.

```bash
git clone https://github.com/abuba8/When-Few-Shot-Hurts.git
cd When-Few-Shot-Hurts

# Recommended: use a fresh virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

LLaMA-3.1-8B and LLaVA-NeXT-Video-7B are gated on Hugging Face. Request access through their model cards, then either export your token or place it in a `.env` file at the repository root:

```bash
export HF_TOKEN="hf_xxx..."
# ── or ──
echo "HF_TOKEN=hf_xxx..." > .env
```

`requirements.txt` (a representative pin set):

```
torch>=2.1
transformers>=4.42
accelerate>=0.30
bitsandbytes>=0.43
sentencepiece
opencv-python
pandas
pyarrow
numpy
tqdm
matplotlib
sentence-transformers
scikit-learn
evaluate
bert_score
huggingface-hub
python-dotenv
jinja2
```

---

## Data preparation

Both benchmarks are publicly available. Place them outside the repository (the scripts use relative paths like `../NExTQA/`):

- **NExT-QA** (Xiao et al., CVPR 2021). 5,440 videos (average 44 s) and 52,044 open-ended questions whose ground-truth answers are often multi-word phrases. Download the open-ended split from the [official release](https://github.com/doc-doc/NExT-QA). The expected layout:
  ```
  NExTQA/
  ├── OE/
  │   ├── train-00000-of-00001.parquet
  │   ├── validation-00000-of-00001.parquet
  │   └── test-00000-of-00001.parquet
  └── videos/
      └── *.mp4
  ```
- **MSVD-QA** (Xu et al., ACM MM 2017). 1,970 short clips (under 10 s) and 50,505 questions whose answers are overwhelmingly single words. Download from the [MSVD-QA repository](https://github.com/xudejing/video-question-answering). The metadata in `src/msvd_qa/{val,test}.json` already follows the `{file_name, question, answer, video_id, id}` schema used by the inference script.

The multi-word (NExT-QA) versus single-word (MSVD-QA) answer contrast is what lets the analysis separate a generation-side format effect from the judge-side label effect.

---

## End-to-end pipeline

The full evaluation grid in the paper covers 80 full-split cells (2 datasets × 2 splits × 2 frame counts × 2 prompting conditions × 5 exemplar sets), plus zero-shot baselines. Below is the recipe for one cell; loop over the axes you care about.

### Step 1 — Frame extraction and dataloader build

The `frame_extraction.py` script is a single CLI with three subcommands:

#### 1a · `extract` — sample frames from videos into per-batch pickles

```bash
python src/frame_extraction.py extract \
    --dataset    nextqa \
    --metadata   ../NExTQA/OE/validation-00000-of-00001.parquet \
    --video-dir  ../NExTQA/videos/ \
    --split      val \
    --n-frames   4 \
    --batch-size 50 \
    --out-dir    ../extracted_feats
```

Each output pickle (`../extracted_feats/val/4_frames/full_feats_val/{i}.pkl`) is a dict of up to `--batch-size` samples; every sample carries the four keys consumed downstream — `video_frames`, `question`, `answer`, `type` — so the per-batch pickles are themselves valid dataloader fragments and an interrupted run only has to redo the unfinished tail.

For MSVD-QA, point `--metadata` at the JSON instead and pass `--dataset msvd`. MSVD-QA does not store a question-type field, so the script infers a coarse What / Who / How / When / Where category from the question's first word — sufficient for the per-category accuracy reported in the aggregation step.

#### 1b · `merge` — concatenate batches into a single dataloader pickle

```bash
python src/frame_extraction.py merge \
    --in-dir   ../extracted_feats/val/4_frames/full_feats_val \
    --out-file ../data_loaders/val/4_frames/dataloader_val_4_frames_1.pkl \
    --start    1 \
    --end      41
```

`merge` re-keys samples with consecutive integers from 1 (the order the inference scripts iterate) and silently drops malformed rows (empty question, missing frames). The summary printout reports raw / valid / invalid counts so corrupted videos are visible.

For very large splits (e.g. NExT-QA `train`) it is convenient to write several dataloader pickles by repeating `merge` over disjoint `--start`..`--end` ranges and shard inference across GPUs — this is what we did to fit the full grid into a single workstation.

#### 1c · `preview` — visual sanity check

```bash
python src/frame_extraction.py preview \
    --in-file     ../data_loaders/val/4_frames/dataloader_val_4_frames_1.pkl \
    --sample-index 1
```

Opens a matplotlib figure with the four (or eight) sampled frames laid out left-to-right and prints the question, ground-truth answer, and category of the chosen sample.

### Step 2 — Prompted generation (LLaVA-NeXT-Video-7B)

The two inference entry points are organised by dataset because the input formats differ — NExT-QA reads from a frame-bundled pickle, MSVD-QA reads videos at runtime — but they share the same model build, deterministic decoding, and resume-from-checkpoint behaviour.

#### 2a · NExT-QA

```bash
python src/next_qa/inference.py \
    --input  ../data_loaders/val/4_frames/dataloader_val_4_frames_1.pkl \
    --output outputs/nextqa/val_4f_zero.json \
    --max-new-tokens 200 \
    --checkpoint-every 10 \
    --resume
```

Output is a JSON dictionary keyed by sample index, with the fields the judge consumes in the next step:

```json
{
  "1": {
    "Question": "...",
    "Original Answer": "...",
    "Generated Answer": "...",
    "Similarity Score": 0.71,
    "BERTScore": 0.86
  }
}
```

`Similarity Score` (sentence-transformers cosine) and `BERTScore` are reported for transparency; the paper's accuracy numbers come from the LLaMA judge at Step 3, since surface metrics fail systematically when the ground truth is a 1–5-word phrase and the VLM emits a multi-sentence answer.

#### 2b · MSVD-QA

```bash
python src/msvd_qa/inference.py \
    --video_dir ../MSVD-QA/videos/ \
    --data_file src/msvd_qa/val.json \
    --output    outputs/msvd/val_4f_zero.json \
    --split     val \
    --n_frames  4 \
    --max-new-tokens 200 \
    --checkpoint-every 10 \
    --resume
```

The MSVD-QA script samples frames lazily on a per-video basis (no pickled dataloader), which is fine because MSVD clips are short (typically < 10 s). For consistent ablation runs, pre-extract once with `frame_extraction.py extract --dataset msvd` and adapt the loader, but for single-cell runs the on-the-fly path is simpler.

Both inference scripts are deterministic (`do_sample=False`) — the greedy output of a given (model, frames, prompt) triple is reproducible bit-for-bit. This is what makes the validation generations byte-identical across prompting conditions and therefore what enables the fixed-generation control central to the label-bias analysis.

#### Switching prompting conditions

The Step 2 scripts produce **zero-shot** outputs. To run **one-shot** or **few-shot**, edit `build_prompt()` in either `inference.py` to inject a `K{i}` exemplar from `src/Prompts/examples/{one,few}_shot.json` into the user content. The exemplar JSON is shipped with the repo so this stays a one-line change; the prompts and the judge templates use the same `K1..K5` namespace so the same selector controls both stages.

### Step 3 — Semantic evaluation (LLaMA-3.1-8B judge)

```bash
python src/Evaluation_pipeline.py \
    outputs/nextqa/val_4f_zero.json \
    outputs/nextqa/val_4f_zero_judged.json \
    --shot-type few \
    --k         K3 \
    --model     meta-llama/Meta-Llama-3.1-8B
```

Arguments:

- `input_json`, `output_json` — positional. The output file is the input plus a `Similarity` field per entry, which is exactly the field the Step 4 aggregators look for.
- `--shot-type` ∈ {zero, one, few}. Selects the rubric template (`Prompts/LLM_as_judge_base.j2` or `Prompts/LLM_as_judge_few.j2`).
- `--k` — the exemplar set used inside the judge prompt (`K1`..`K5`). The exemplar file is the same one consumed at Step 2; the same `--k` should be used at both steps for the cell to correspond to a paper row.
- `--hf-token` defaults to `$HF_TOKEN`.

The judge runs in 4-bit `nf4` quantisation with double quantisation. The decision token is the higher-probability logit between `Yes` and `No`; unparseable (non-Yes/No) outputs are mapped to `No` under the conservative policy reported in the paper, and a drop-unparseable sensitivity analysis preserves all within-condition rankings.

> **Reproducing the label-bias result.** Setting `--k K3` with `--shot-type few` is the acceptance-skewed (Yes-fraction 0.75) condition that produces the inflation spike. Compare it against `--k K2` (rejection-skewed, 0.25) and `--k K4`/`--k K5` (balanced, 0.50) on the same fixed Step-2 outputs to reproduce the directional, one-sided departure from human consensus.

### Step 4 — Accuracy aggregation

`scripts/compute_accuracy.py` is a single CLI that handles both datasets. It accepts one or more judge JSONs (sharded inference runs are merged with file-index suffixes on key collisions) and reports the judge's overall acceptance rate plus a per-category breakdown matched to the dataset.

> **Read these as judge verdict (acceptance) rates, not as accuracy against human ground truth.** Judge–human agreement is only fair across the grid (κ = 0.17–0.36, well below the ≈0.69 inter-annotator agreement), so absolute numbers are best interpreted as relative comparisons across cells. The S3 few-shot cells in particular are the label-bias artifact and must not be read as system accuracy.

#### NExT-QA — overall + fine-grained type + broad category

The NExT-QA breakdown pulls each entry's type code (CW / CH / TN / TC / TP / DC / DL / DO / DB) from the parquet by matching on the question text, then groups them into Causal / Temporal / Descriptive.

```bash
python scripts/compute_accuracy.py nextqa \
    outputs/nextqa/val_4f_zero_judged.json \
    --parquet ../NExTQA/OE/validation-00000-of-00001.parquet
```

#### MSVD-QA — overall + per-Wh-word breakdown

MSVD-QA does not store a question-type field, so the breakdown is inferred from the question's first word (What / Who / How / When / Where) — the same convention used by `frame_extraction.py` when building the MSVD dataloader.

```bash
python scripts/compute_accuracy.py msvd outputs/msvd/val_4f_zero_judged.json
```

Pass `--no-breakdown` for either dataset to print overall accuracy only, and pass multiple JSON paths positionally to merge sharded runs:

```bash
python scripts/compute_accuracy.py msvd \
    outputs/msvd/val_4f_zero_shard1.json \
    outputs/msvd/val_4f_zero_shard2.json \
    --no-breakdown
```

---

## Reproducing the paper's tables

The main results table (80 full-split cells plus zero-shot baselines) is produced by running the four-step pipeline above for every combination of (dataset, split, frame count, condition, set):

```text
datasets       = {NExT-QA, MSVD-QA}
splits         = {val, test}
frames         = {4, 8}
conditions     = {zero, one, few}     # zero is run once per (dataset, split, frames)
sets           = {K1, K2, K3, K4, K5} # exemplar sets, paper labels S1..S5
```

Three practical notes:

- **The label-bias result exploits a cost asymmetry.** Step 2 (LLaVA generation) only depends on the **generator-side** exemplar; varying the **judge-side** exemplar at Step 3 reuses the same Step-2 outputs. On the 200-example stratified validation sample, greedy decoding makes the Step-2 outputs identical across prompting conditions, so any condition-dependent variation observed there is a judge-side effect. This is the fixed-generation control that isolates the label-bias mechanism.
- **Frame count is inert.** Every 4-frame and 8-frame accuracy pair agrees to within about one point, ruling out frame sampling as a factor in any comparison.
- **S1 is excluded from the verdict-balance analysis** (it predates the verdict-balance control). It is retained in the exemplar-variance and one-shot analyses. The S3 few-shot cells are flagged as the label-bias artifact and reported separately from the mean/std of the non-skewed sets.

A reproducer shell loop is provided in `scripts/run_grid.sh`. Adapt the GPU IDs and paths before running.

---

## Human-annotation protocol

For each dataset we sample 200 test examples stratified by question type. Two graduate-level annotators (A1 stricter, A2 more lenient), both blinded to the judge's decisions, independently label each (question, ground truth, generated answer) triple under the three-class rubric (Direct Match / Indirect Context / Different Context, collapsed to accept / reject). Where A1 and A2 disagreed, **Claude Opus 4.7** was used as an automatic, independent third rater under the same rubric to break ties.

Inter-annotator agreement was substantial (Landis and Koch, 1977) and nearly identical across datasets:

| Dataset | A1 vs A2 (raw) | A1 vs A2 (κ) | Opus vs A1 (κ) | Opus vs A2 (κ) |
| --- | --- | --- | --- | --- |
| NExT-QA | 85.00% | 0.693 | 0.571 | 0.791 |
| MSVD-QA | 86.00% | 0.695 | 0.550 | 0.691 |

The tiebreaker resolved 30/200 disagreements on NExT-QA and 28/200 on MSVD-QA, siding with the more lenient annotator (A2) more often than A1 on both datasets. Because any leniency it shares would raise the consensus acceptance rate, its influence is **conservative** for the inflation effect we report. The consensus acceptance rates used as the reference throughout are 56.5% (NExT-QA) and 63.5% (MSVD-QA). The full annotation files, including per-example A1/A2/Opus labels, are released under `annotations/` so that the κ numbers above are directly recomputable.

The question-category composition of the 200-example, type-stratified validation samples:

| Dataset | Category | Val. n |
| --- | --- | --- |
| NExT-QA | Causal | 105 |
| NExT-QA | Temporal | 62 |
| NExT-QA | Descriptive | 33 |
| MSVD-QA | What | 119 |
| MSVD-QA | Who | 74 |
| MSVD-QA | How | 6 |
| MSVD-QA | When | 1 |
| MSVD-QA | Where | 0 |

The MSVD-QA sample under-represents How and When and contains no Where — a coverage caveat carried forward in the limitations.

---

## Hardware and runtime

Compute: 3 × NVIDIA A30 24 GB, roughly 3 GPU-hours per split. Both models are loaded in 4-bit (`nf4` with double quantisation) via `bitsandbytes`, so a single 24 GB GPU is sufficient to run any individual cell. Greedy decoding is used throughout, with validation generations byte-identical across prompting conditions. For context, a full fine-tune in this environment was projected at more than 140 GPU-hours per epoch — the compute gap that keeps the training-free, judge-scored population growing.

---

## Limitations

The findings are framed conservatively in the paper; for completeness:

- **The asymmetry rests on a single rejection-skewed set per dataset.** The rejection-side conclusion is corroborated by that set's balanced error profile, its highest κ, and the one-shot (k=1) replication — not by point estimates alone — but it is not yet traced as a full curve.
- **The acceptance-skewed set (S3) also differs in category composition** (1/2/1 vs the standard 2/1/1). Three arguments from data already in the paper localize the effect to verdict balance rather than category: the fixed-generation control holds the generated text fixed; judge accuracy varies only ~9 points across the three NExT-QA categories (too little to produce a 20–30-point spike); and the one-shot acceptance-direction effect holds for S5, a standard-category set. A category-matched skewed set remains the cleanest follow-up.
- **The saturating-prior mechanism is an interpretation, not a direct measurement.** Confirming it would require logging the judge's per-set Yes/No logit margin, which was not recorded in these runs and is left to future work.
- **The Yes-fraction correlation is single-condition, not a dose–response.** Pearson r between demonstration Yes-fraction and judge acceptance rate is 0.76 (NExT-QA) / 0.79 (MSVD-QA) at n = 4, but the statistic is carried entirely by S3: removing it reverses the correlation to r = −0.99 (NExT-QA) and collapses it to r = −0.06 (MSVD-QA). The effect is a single-condition jump, not a smooth trend.
- **Exemplar variance is a first-order estimate, not a confidence interval.** The within-condition std is 5.6–7.2 points; the NExT-QA one-shot lift over zero-shot at n = 5 is +5.88 with t = 1.93, p = 0.126.
- **Judge agreement is fair, not substantial** (κ = 0.17–0.36 versus ≈0.69 between human annotators), so absolute accuracies should be read as relative comparisons across cells.
- **One backbone, two datasets, single runs per cell.** Effects under LLaVA-NeXT-Video-7B + LLaMA-3.1-8B may not transfer to larger or better-calibrated judges. Stronger judges would plausibly attenuate the label-bias effect but, given the existing LLM-judge-bias evidence, are unlikely to eliminate it.
- **MSVD-QA validation under-represents How (n = 6) and When (n = 1) and contains no Where.** Minority-category κ values should not be over-interpreted.

---

## Citation

If you use this code, the prompts, the exemplar sets, or the human annotations, please cite:

```bibtex
@inproceedings{yourjudge2026,
  title     = {Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric},
  author    = {Anonymous},
  booktitle = {Proceedings of the EMNLP Workshop},
  year      = {2026}
}
```

*(Author and venue details will be de-anonymized in the camera-ready version.)*

---

## License and acknowledgements

Code in this repository is released under the MIT License. This work uses the following pre-trained models and datasets, each governed by their own licenses:

- [LLaVA-NeXT-Video-7B-hf](https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf) (Liu et al., 2024)
- [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) (Meta AI, 2024) — Llama 3.1 Community License
- [NExT-QA](https://github.com/doc-doc/NExT-QA) (Xiao et al., CVPR 2021)
- [MSVD-QA](https://github.com/xudejing/video-question-answering) (Xu et al., ACM MM 2017)
