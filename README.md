# Hier_Encoder

> **Central Research Question:**  
> Do hierarchical sequence encoders improve next-item recommendation performance compared with standard sequential recommendation models under a unified and fair benchmark protocol?

Hier_Encoder is a research-oriented benchmark framework for sequential recommendation using the Amazon Review Dataset 2023.

This project focuses on benchmarking standard sequential recommendation architectures against their hierarchical variants for next-item recommendation tasks.

The framework is designed for reproducible research, fair model comparison, and modular experimentation.

---

# Table of Contents

1. [Overview](#overview)
2. [Supported Models](#supported-models)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)
6. [Installation](#installation)
7. [Dataset Preparation](#dataset-preparation)
8. [Configuration System](#configuration-system)
9. [Recommended Workflow](#recommended-workflow)
10. [Training](#training)
11. [Testing](#testing)
12. [Benchmark Results](#benchmark-results)
13. [Evaluation Metrics](#evaluation-metrics)
14. [Fairness Protocol](#fairness-protocol)
15. [Reproducibility](#reproducibility)
16. [Future Work](#future-work)
17. [License](#license)

---

# Overview

Hier_Encoder implements multiple sequential recommendation architectures and their hierarchical extensions under a unified benchmark framework.

The project supports:

- modular architecture
- reproducible experiments
- fairness-aware benchmarking
- automatic dataset preprocessing
- checkpoint management
- benchmark result aggregation
- multi-model experimentation
- standardized evaluation

Current benchmark setting:

- Dataset: Amazon Review Dataset 2023
- Domain: Cell Phones & Accessories
- Task: Next-item recommendation
- Evaluation: HR@K, NDCG@K

---

# Supported Models

## Baseline Sequential Models

| Model | Architecture Type | Description |
|------|------------------|-------------|
| GRU4Rec | RNN-based | GRU sequential recommendation model |
| SASRec | Transformer-based | Self-attention sequential recommendation |
| NextItNet | CNN-based | Dilated convolution sequential recommendation |

---

## Hierarchical Models

| Model | Architecture Type | Description |
|------|------------------|-------------|
| HierGRU | Hierarchical RNN | Multi-scale temporal GRU encoder |
| HierSASRec | Hierarchical Transformer | Hierarchical self-attention sequence modeling |
| HierNextItNet | Hierarchical CNN | Multi-scale dilated convolution encoder |

---

# System Architecture

Hier_Encoder follows a modular benchmark pipeline.

## High-Level Workflow

```text
Raw Dataset
    ↓
Download
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
```

---

## Detailed Pipeline

```text
Amazon Review Dataset 2023
(data/raw)
    │
    ├── Cell_Phones_and_Accessories.jsonl.gz
    └── meta_Cell_Phones_and_Accessories.jsonl.gz
    │
    ▼
scripts/download_data.py
    │
    ▼
datasets/preprocess.py
    │
    ├── JSON parsing
    ├── invalid row filtering
    ├── k-core filtering
    ├── user encoding
    ├── item encoding
    ├── temporal bucket generation
    ├── sequence construction
    └── train / validation / test split
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
    └── processed dataset loading
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
│   ├── download_data.py
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
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Requirements

Recommended environment:

| Component | Recommended Version |
|----------|--------------------|
| Python | 3.10+ |
| PyTorch | 2.x |
| CUDA | 11.8+ |
| RAM | 16 GB+ |
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

### GPU Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### CPU Installation

```bash
pip install torch torchvision torchaudio
```

---

### Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

This project uses Amazon Review Dataset 2023.

Dataset domain:

```text
Cell Phones & Accessories
```

---

## Step 1: Download Dataset

Automatically download raw dataset files:

```bash
python scripts/download_data.py
```

Downloaded files:

```bash
data/raw/
├── Cell_Phones_and_Accessories.jsonl.gz
└── meta_Cell_Phones_and_Accessories.jsonl.gz
```

---

## Step 2: Preprocess Dataset

Convert raw files into benchmark-ready processed data:

```bash
python scripts/preprocess.py
```

Generated files:

```bash
data/processed/
├── train.pkl
├── val.pkl
├── test.pkl
├── user_encoder.pkl
└── item_encoder.pkl
```

---

# Configuration System

The project uses layered configuration.

Priority:

```text
Model Config
    ↓
Fairness Config
    ↓
Base Config
```

---

## base.yaml

Shared infrastructure settings.

Controls:

- dataset paths
- logging
- evaluation
- reproducibility
- shared defaults

---

## fairness.yaml

Benchmark fairness constraints.

Shared across all models:

- embedding dimension
- batch size
- learning rate
- optimizer
- epochs
- early stopping
- evaluation settings

---

## model configs

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

Controls architecture-specific settings:

- hidden size
- dropout
- transformer heads
- CNN kernel size
- GRU layers

---

# Recommended Workflow

## Step 1

Download dataset:

```bash
python scripts/download_data.py
```

---

## Step 2

Preprocess dataset:

```bash
python scripts/preprocess.py
```

---

## Step 3

Train a model:

```bash
python main.py --mode train --model sasrec
```

---

## Step 4

Test trained model:

```bash
python main.py --mode test --model sasrec
```

---

## Step 5

Compare benchmark results:

```bash
results/tables/benchmark_results.csv
```

---

# Training

## Train Single Model

GRU4Rec:

```bash
python main.py --mode train --model gru4rec
```

SASRec:

```bash
python main.py --mode train --model sasrec
```

HierSASRec:

```bash
python main.py --mode train --model hier_sasrec
```

---

## Train All Models

```bash
python main.py --mode train --model all
```

Models trained:

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

## Checkpoints

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

## Benchmark Table

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
| gru4rec | 0.12 | 0.08 | 0.21 | 0.11 | 0.31 | 0.15 |
| sasrec | 0.15 | 0.10 | 0.25 | 0.14 | 0.37 | 0.19 |
| hier_sasrec | 0.17 | 0.11 | 0.28 | 0.16 | 0.40 | 0.21 |

---

# Evaluation Metrics

Implemented metrics:

| Metric | Description |
|--------|-------------|
| HR@K | Hit Rate |
| NDCG@K | Normalized Discounted Cumulative Gain |

Supported:

- HR@5
- HR@10
- HR@20
- NDCG@5
- NDCG@10
- NDCG@20

---

# Fairness Protocol

All benchmark comparisons follow the same constraints:

- identical embedding dimension
- identical optimizer
- identical learning rate
- identical batch size
- identical evaluation metrics
- identical train / validation / test split
- identical early stopping protocol

Configuration:

```bash
configs/fairness.yaml
```

---

# Reproducibility

Controlled via:

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

Potential extensions:

- BERT4Rec
- Contrastive Sequential Recommendation
- Multi-behavior Recommendation
- Session-based Recommendation
- Self-supervised Sequence Modeling
- Hyperparameter Search
- Distributed Training
- Docker Deployment

---

# License

This repository is intended for academic and research use.
