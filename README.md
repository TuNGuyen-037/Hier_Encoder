# Hier_Encoder

> **Central Research Question:**  
> Do hierarchical sequence encoders improve next-item recommendation performance compared with standard sequential recommendation models under a unified and fair benchmark protocol?

Hier_Encoder is a research-oriented benchmark framework for sequential recommendation using Amazon Review Dataset 2023.

The project focuses on comparing standard sequential recommendation architectures with their hierarchical variants for modeling short-term and long-term user behavior.

---

# Table of Contents

1. [Overview](#overview)
2. [Supported Models](#supported-models)
3. [Architecture Pipeline](#architecture-pipeline)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Dataset Preparation](#dataset-preparation)
8. [Configuration System](#configuration-system)
9. [Training](#training)
10. [Testing](#testing)
11. [Benchmark Results](#benchmark-results)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Fairness Protocol](#fairness-protocol)
14. [Future Work](#future-work)

---

# Overview

This project implements multiple sequential recommendation architectures and their hierarchical extensions for next-item recommendation tasks.

The framework supports:

- unified training pipeline
- fairness-aware benchmarking
- train / validation / test evaluation
- automatic checkpoint saving
- benchmark aggregation
- multi-model training
- reproducibility control

The current implementation is built on Amazon Review Dataset 2023 using:

- user-item interaction sequences
- timestamp-aware sequence encoding
- hierarchical temporal representations

---
# System Architecture

Hier_Encoder follows a modular benchmark architecture for sequential recommendation research.

The workflow is:

```text
Raw Dataset
    ↓
Preprocessing
    ↓
Processed Dataset
    ↓
Dataset Loader
    ↓
Model
    ↓
Trainer
    ↓
Evaluation
    ↓
Benchmark Results
Amazon 2023 Raw Dataset
(data/raw)
    │
    ├── Cell_Phones_and_Accessories.jsonl.gz
    └── meta_Cell_Phones_and_Accessories.jsonl.gz
    │
    ▼
datasets/preprocess.py
    │
    ├── JSON parsing
    ├── data cleaning
    ├── k-core filtering
    ├── user/item encoding
    ├── sequence generation
    ├── temporal bucket construction
    └── train/val/test splitting
    │
    ▼
Processed Dataset
(data/processed)
    │
    ├── train.pkl
    ├── val.pkl
    ├── test.pkl
    ├── user_encoder.pkl
    └── item_encoder.pkl
    │
    ▼
datasets/amz2023.py
    │
    └── load processed benchmark data
    │
    ▼
datasets/dataloader.py
    │
    ├── train_loader
    ├── val_loader
    └── test_loader
    │
    ▼
models/
    │
    ├── GRU4Rec
    ├── HierGRU
    ├── SASRec
    ├── HierSASRec
    ├── NextItNet
    └── HierNextItNet
    │
    ▼
trainers/trainer.py
    │
    ├── forward pass
    ├── loss computation
    ├── optimization
    ├── validation
    ├── early stopping
    └── checkpoint saving
    │
    ▼
evaluation/
    │
    ├── HR@K
    └── NDCG@K
    │
    ▼
results/

# Supported Models

## Baseline Sequential Models

| Model | Type | Description |
|------|------|-------------|
| GRU4Rec | RNN-based | GRU sequential recommender |
| SASRec | Transformer-based | Self-attention sequential recommendation |
| NextItNet | CNN-based | Dilated convolution sequential recommendation |

---

## Hierarchical Models

| Model | Type | Description |
|------|------|-------------|
| HierGRU | Hierarchical RNN | Multi-scale temporal GRU encoder |
| HierSASRec | Hierarchical Transformer | Hierarchical self-attention sequence modeling |
| HierNextItNet | Hierarchical CNN | Hierarchical dilated convolution encoder |

---

# Architecture Pipeline

## Overall Pipeline

```text
┌─────────────────────────────────────────────────────┐
│                 AMAZON REVIEW DATASET               │
│                                                     │
│  Cell Phones & Accessories                          │
│  ├── review interactions                            │
│  ├── timestamps                                     │
│  └── metadata                                       │
└──────────────────────────┬──────────────────────────┘
                           │
                    datasets/preprocess.py
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│              DATA PREPROCESSING                     │
│                                                     │
│  • read jsonl.gz                                    │
│  • remove invalid rows                              │
│  • k-core filtering                                 │
│  • user/item encoding                               │
│  • sequence generation                              │
│  • time bucket construction                         │
│  • train/val/test split                             │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                  DATALOADER                         │
│                                                     │
│  train_loader                                       │
│  val_loader                                         │
│  test_loader                                        │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                     TRAINER                         │
│                                                     │
│  • training loop                                    │
│  • evaluation                                       │
│  • early stopping                                   │
│  • checkpoint saving                                │
│  • benchmark logging                                │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                   EVALUATION                        │
│                                                     │
│  • HR@K                                             │
│  • NDCG@K                                           │
│  • benchmark aggregation                            │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
               results/tables/benchmark_results.csv
```

---

# Project Structure

```bash
Hier_Encoder/
│
├── configs/
│   ├── base.yaml
│   ├── fairness.yaml
│   └── model/
│       ├── gru4rec.yaml
│       ├── hier_gru.yaml
│       ├── sasrec.yaml
│       ├── hier_sasrec.yaml
│       ├── nextitnet.yaml
│       └── hier_nextitnet.yaml
│
├── datasets/
│   ├── __init__.py
│   ├── amz2023.py
│   ├── dataloader.py
│   └── preprocess.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   └── metrics.py
│
├── losses/
│   ├── __init__.py
│   ├── cross_entropy.py
│   └── ranking_loss.py
│
├── models/
│   ├── __init__.py
│   ├── gru4rec.py
│   ├── hier_gru.py
│   ├── sasrec.py
│   ├── hier_sasrec.py
│   ├── nextitnet.py
│   └── hier_nextitnet.py
│
├── results/
│   ├── __init__.py
│   ├── aggregate.py
│   ├── checkpoints/
│   ├── logs/
│   └── tables/
│
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── test.py
│
├── trainers/
│   ├── __init__.py
│   └── trainer.py
│
├── utils/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── device.py
│   ├── io.py
│   └── seed.py
│
├── data/
│   ├── raw/
│   │   ├── Cell_Phones_and_Accessories.jsonl.gz
│   │   └── meta_Cell_Phones_and_Accessories.jsonl.gz
│   │
│   └── processed/
│       ├── train.pkl
│       ├── val.pkl
│       ├── test.pkl
│       ├── user_encoder.pkl
│       └── item_encoder.pkl
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```
---

# Requirements

| Component | Recommended |
|-----------|-------------|
| Python | 3.10+ |
| PyTorch | 2.x |
| CUDA | 11.8+ |
| RAM | 16 GB |
| GPU VRAM | 4 GB+ |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/TuNGuyen-037/Hier_Encoder.git
cd Hier_Encoder
```

---

## Install Dependencies

### GPU Version

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CPU Version

```bash
pip install torch torchvision torchaudio
```

### Install Remaining Packages

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

This project uses Amazon Review Dataset 2023.

---

## Download Review Data

Cell Phones & Accessories:

```text
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Cell_Phones_and_Accessories.jsonl.gz
```

---

## Download Meta Data

```text
https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Cell_Phones_and_Accessories.jsonl.gz
```

---

## Place Files

Put both files inside:

```bash
data/
```

Final structure:

```bash
data/
├── Cell_Phones_and_Accessories.jsonl.gz
└── meta_Cell_Phones_and_Accessories.jsonl.gz
```

---

# Configuration System

The framework uses layered configuration.

Priority order:

```text
Model Config
    ↓
Fairness Config
    ↓
Base Config
```

---

## base.yaml

Shared infrastructure configuration.

Controls:

- dataset paths
- logging
- reproducibility
- evaluation
- default training settings

---

## fairness.yaml

Defines fair benchmark constraints shared by all models.

Includes:

- embedding dimension
- batch size
- learning rate
- optimizer
- epochs
- early stopping

This ensures fair comparison across all architectures.

---

## Model Configs

Located in:

```bash
configs/model/
```

Examples:

```bash
gru4rec.yaml
sasrec.yaml
hier_sasrec.yaml
```

Controls:

- hidden size
- dropout
- attention heads
- number of layers
- kernel size

---

# Training

## Train Single Model

### GRU4Rec

```bash
python main.py --mode train --model gru4rec
```

---

### SASRec

```bash
python main.py --mode train --model sasrec
```

---

### HierSASRec

```bash
python main.py --mode train --model hier_sasrec
```

---

## Train All Models

```bash
python main.py --mode train --model all
```

This automatically trains:

- GRU4Rec
- HierGRU
- SASRec
- HierSASRec
- NextItNet
- HierNextItNet

---

# Testing

## Test Single Model

```bash
python main.py --mode test --model sasrec
```

---

## Test Hierarchical Model

```bash
python main.py --mode test --model hier_nextitnet
```

---

## Test All Models

```bash
python main.py --mode test --model all
```

---

# Benchmark Results

## Model Checkpoints

Saved in:

```bash
results/checkpoints/
```

Examples:

```bash
best_gru4rec.pt
best_sasrec.pt
best_hier_sasrec.pt
```

---

## Benchmark Tables

Saved in:

```bash
results/tables/
```

Generated file:

```bash
benchmark_results.csv
```

Example:

| model | hr@5 | ndcg@5 | hr@10 | ndcg@10 | hr@20 | ndcg@20 |
|------|------|---------|--------|----------|--------|----------|

---

# Evaluation Metrics

Implemented metrics:

| Metric | Description |
|--------|-------------|
| HR@K | Hit Rate |
| NDCG@K | Normalized Discounted Cumulative Gain |

Supported K values:

- @5
- @10
- @20

---

# Fairness Protocol

To ensure fair comparison:

- same embedding dimension
- same optimizer
- same batch size
- same learning rate
- same evaluation protocol
- same train/validation/test split
- same metrics

All shared settings are defined in:

```bash
configs/fairness.yaml
```

---

# Reproducibility

Controlled through:

```yaml
reproducibility:
  seeds: [42]
  deterministic: false
```

Located in:

```bash
configs/base.yaml
```

---

# Future Work

Potential future extensions:

- BERT4Rec
- Contrastive Sequential Recommendation
- Multi-behavior Recommendation
- Session-based Recommendation
- Self-supervised Sequence Modeling
- Hyperparameter Search
- Distributed Training
- Docker Deployment

---

# Notes

Current framework supports:

- next-item recommendation
- sequential recommendation
- hierarchical sequence modeling
- benchmark comparison
- automatic evaluation
- checkpoint management
- benchmark aggregation

---

# License

This repository is intended for academic and research purposes.
```
