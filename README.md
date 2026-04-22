# MASKED TRANSFORMER MODELS FOR SYMBOLIC AUTOMATIC CHORD RECOGNITION VIA SYNTHETIC DATA AUGMENTATION (2026)

Anonymous repository for the paper:

**MASKED TRANSFORMER MODELS FOR SYMBOLIC AUTOMATIC CHORD RECOGNITION VIA SYNTHETIC DATA AUGMENTATION**

This repository contains the code, evaluation scripts, and reproduction material associated with the submitted paper.

---

# Overview

Automatic chord recognition (ACR) aims to estimate time-aligned chord labels from musical input.

While most prior work has focused on audio recordings, this paper studies **symbolic automatic chord recognition** using MIDI / MusicXML representations.

The proposed framework combines:

- masked Transformer sequence modeling  
- symbolic score representations  
- synthetic dataset generation  
- cross-dataset evaluation  

---

# Main Contributions

- Adapts masked encoder-only Transformer models to symbolic ACR  
- Uses note evidence from full symbolic scores (melody + accompaniment)  
- Introduces synthetic symbolic data augmentation from melody-harmony corpora  
- Improves generalization under out-of-distribution evaluation  
- Outperforms recent symbolic baselines on multiple metrics  

---

# Repository Contents

## Core Files

### `train_semh.py`

Training script for the proposed model.

Includes:

- dataset loading  
- curriculum masking  
- optimization  
- checkpoint saving  

---

### `generate_order_test.py`

Inference script used to generate chord predictions from trained checkpoints.

Includes iterative masked decoding strategies.

---

### `models.py`

Contains neural network architectures.

Main model used in the paper:

- `SEModular`

Single-encoder Transformer with:

- score projection layers  
- chord token embeddings  
- positional encoding  
- masked token prediction head  

---

### `GridMLM_tokenizers_old.py`

Tokenizer / symbolic representation pipeline.

Converts MIDI / MusicXML into:

- quarter-note grids  
- pitch-class vectors  
- bar markers  
- chord token sequences  

Main configuration used in the paper:

- `Q4_L80_bar_PC`

---

### `data_utils.py`

Dataset loading and batching utilities.

---

### `train_utils.py`

Training utilities:

- masking schedules  
- loss functions  
- logging  
- metrics helpers  

---

### `music_utils.py`

Music processing helpers:

- transposition  
- symbolic score handling  
- utility functions  

---

### `p_values.py`

Statistical significance testing.

Includes:

- paired t-test  
- Wilcoxon signed-rank test  

Outputs summary `.csv` tables.

---

# Dataset Sources

## Synthetic Training Sources

Constructed from publicly available melody-harmony corpora:

- HookTheory  
- Nottingham Dataset  
- Wikifonia (archived public sources)

---

## Evaluation Benchmark

Out-of-distribution evaluation uses the **Chord Melody Dataset**.

Public source:

https://https://github.com/shiehn/chord-melody-dataset

---

# Baseline Systems

## BACHI

Public repository:

https://github.com/AndyWeasley2004/BACHI_Chord_Recognition

## AugmentedNet

Public repository:

https://github.com/napulen/AugmentedNet

---

# Prior Model Sources

This work builds upon earlier symbolic harmonization research:

## Encoder-Only Transformers for Melodic Harmonization

https://proceedings.mlr.press/v303/kaliakatsos-papakostas26a.html

## Diffusion-inspired Masked Language Modeling for Symbolic Harmony Generation

https://www.mdpi.com/2076-3417/15/17/9513

---

# Installation

```bash
pip install -r requirements.txt
```

Recommended:

```text
Python 3.10+
PyTorch compatible CUDA or CPU setup
```

---

# Example Training

```bash
python train_semh.py -m SE -c f2f -f Q4_L80_bar_PC -u 0
```

---

# Example Inference

```bash
python generate_order_test.py -m SE -c f2f -f Q4_L80_bar_PC -u 0 --modelpath saved_models/SE/Q4_L80_bar_PC/f2f.pt --input test_data/input --output test_data/output
```

---

# Statistical Tests

```bash
python p_values.py
```

Produces:

- significance summaries  
- pairwise model comparisons  
- csv tables for paper reporting  

---

# Notes

This repository is released anonymously for peer-review and reproducibility purposes.

Some external dataset links may change over time. Public mirrors or archived sources may be required.
