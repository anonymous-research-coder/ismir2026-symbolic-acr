# Synthetic Data and Masked Transformers for Symbolic Automatic Chord Recognition

Official repository for the paper:

**Synthetic Data and Masked Transformers for Symbolic Automatic Chord Recognition**

This repository contains the code used to train, evaluate, and reproduce the experiments presented in the paper.

---

# Overview

Automatic chord recognition (ACR) aims to estimate time-aligned chord labels from musical input.  
While most prior work has focused on audio recordings, this work studies **symbolic automatic chord recognition** using MIDI / MusicXML representations.

We propose:

- A masked **single-encoder Transformer** for symbolic chord recognition
- Transfer learning from prior symbolic harmonization models
- A synthetic multi-source dataset generation pipeline
- Cross-dataset evaluation on an out-of-distribution benchmark

---

# Main Contributions

- Adapts an encoder-only masked Transformer from melodic harmonization to symbolic ACR
- Uses full symbolic score evidence (melody + accompaniment)
- Introduces synthetic symbolic training corpus construction
- Improves performance over recent symbolic baselines
- Demonstrates improved generalization under out-of-distribution evaluation

---

# Repository Structure

## Core Files

### `train_semh.py`

Training script for the proposed **SESACR** model.

Used for:

- loading tokenized datasets
- curriculum masking training
- optimization
- checkpoint saving

---

### `generate_order_test.py`

Inference / generation script.

Used to run trained models on unseen symbolic files and generate predicted chord outputs.

Supports iterative masked decoding.

---

### `models.py`

Contains neural network architectures.

Main model used in the paper:

- `SEModular`

Single-encoder Transformer with:

- score input projection
- chord token embeddings
- positional encoding
- masked chord prediction head

---

### `GridMLM_tokenizers_old.py`

Tokenizer and symbolic representation pipeline.

Converts MIDI / MusicXML into:

- quarter-note grid
- pitch-class vectors
- bar markers
- chord token sequences

Configuration used in paper:

- `Q4_L80_bar_PC`

---

### `data_utils.py`

Dataset loading utilities.

Includes batching / collate functions.

---

### `train_utils.py`

Training utilities:

- masking schedules
- losses
- curriculum helpers
- metrics logging

---

### `music_utils.py`

Music processing helpers:

- transposition
- score conversion
- symbolic processing utilities

---

### `p_values.py`

Runs paired statistical significance tests used in paper:

- paired t-test
- Wilcoxon signed-rank test

Outputs `.csv` summaries.

---

# Dataset Sources

## Training Corpora

Synthetic corpus built from:

### HookTheory

https://www.hooktheory.com/

### Nottingham Dataset

https://ifdo.ca/~seymour/nottingham/nottingham.html

### Wikifonia (archived community mirrors)

Various public mirrors exist.

---

## Evaluation Benchmark

### Chord Melody Dataset

Used as out-of-distribution evaluation benchmark.

Public source:

https://github.com/wayne391/ChordMelodyDataset

---

# Baseline Systems

## BACHI

Repository:

https://github.com/ptnghia-j/BACHI

Used as symbolic ACR baseline.

---

## AugmentedNet

Repository:

https://github.com/napulen/AugmentedNet

Used as symbolic harmonic analysis baseline.

---

# Original Model Source

The proposed model is adapted from prior melodic harmonization work:

### Encoder-Only Transformers for Melodic Harmonization

Paper:

https://proceedings.mlr.press/v303/kaliakatsos-papakostas26a.html

### Diffusion-inspired Masked Language Modeling for Symbolic Harmony Generation

Paper:

https://www.mdpi.com/2076-3417/15/17/9513

---

# Installation

```bash
pip install -r requirements.txt
