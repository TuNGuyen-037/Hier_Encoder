# Hier_Encoder

> **Central Research Question:** > Do hierarchical sequence encoders improve next-item recommendation performance compared with standard sequential recommendation models under a unified and fair benchmark protocol?

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
| HierGNN + GRU4Rec | Hierarchical GNN & RNN | HierGNN enhanced sequential GRU |
| HierGNN + SASRec | Hierarchical GNN & Transformer | HierGNN enhanced self-attention sequence modeling |
| HierGNN + NextItNet | Hierarchical GNN & CNN | HierGNN enhanced dilated convolution encoder |

---

# System Architecture

Hier_Encoder follows a modular benchmark pipeline.

## High-Level Workflow

#text
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


---Detailed Pipeline

#Amazon Review Dataset 2023
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
    ├── JSON parsing (Reviews & Metadata)
    ├── invalid row filtering
    ├── k-core filtering
    ├── user encoding
    ├── item encoding
    ├── temporal bucket generation
    ├── hierarchical taxonomy graph construction
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
    ├── item_encoder.pkl
    ├── item_taxonomy.pkl
    └── graph_edges.pkl
    │
    ▼
datasets/amz2023.py
    │
    └── processed dataset loading & category paths mapping
    │
    ▼
datasets/dataloader.py
    │
    ├── train_loader (seq, time_seq, category_paths, target)
    ├── val_loader (seq, time_seq, category_paths, target)
    └── test_loader (seq, time_seq, category_paths, target)
    │
    ▼
models/
    │
    ├── GRU4Rec
    ├── SASRec
    ├── NextItNet
    └── hier_gnn.py (Graph Encoder + Hierarchical Encoder + Fusion)
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



---Project Structure
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
│   ├── sasrec.py
│   ├── nextitnet.py
│   └── hier_gnn.py
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

Requirements
Recommended environment:
-----------------------------------------------------------
Component                     |   Recommended Version      |
Python                        |         3.10+              |
PyTorch                       |         2.2.2              |
CUDA                          |         11.8+ / 12.1+      |
RAM                           |         16 GB+GPU          |
VRAM                          |         4 GB+              |
-----------------------------------------------------------

---Installation
Clone Repository: 
git clone [https://github.com/TuNGuyen-037/Hier_Encoder.git](https://github.com/TuNGuyen-037/Hier_Encoder.git)
cd Hier_Encoder

---Install Dependencies
GPU Installation:
pip install torch==2.2.2 torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

---CPU Installation:
pip install torch==2.2.2 torchvision torchaudio


---Install Remaining Dependencies: 
pip install -r requirements.txt

---Dataset Preparation
This project uses Amazon Review Dataset 2023.
Dataset domain: Cell Phones & Accessories
Step 1: Download DatasetAutomatically download raw dataset files: python scripts/download_data.py
Downloaded files:data/raw/
├── Cell_Phones_and_Accessories.jsonl.gz
└── meta_Cell_Phones_and_Accessories.jsonl.gz
Step 2: Preprocess DatasetConvert raw files into benchmark-ready processed data:python scripts/preprocess.py
Generated files:data/processed/
├── train.pkl
├── val.pkl
├── test.pkl
├── user_encoder.pkl
├── item_encoder.pkl
├── item_taxonomy.pkl
└── graph_edges.pkl

---Configuration System
The project uses layered configuration.
Priority:
Model Config
    ↓
Fairness Config
    ↓
Base Config

---base.yaml
Shared infrastructure settings.
Controls:
- dataset paths
- logging
- evaluation
- reproducibility
- shared defaults
---fairness.yaml
Benchmark fairness constraints.
Shared across all models:
- embedding dimension
- batch size
- learning rate
- optimizer
- epochsearly stopping
- evaluation settings

---model configs
Located in:
configs/model/

---Examples:
Bashgru4rec.yaml
sasrec.yaml
hier_sasrec.yaml

---Controls architecture-specific settings:
- hidden size
- dropout
- transformer heads
- CNN kernel size
- GRU layers

---Recommended Workflow
- Step 1. Download dataset:python scripts/download_data.py
- Step 2. Preprocess dataset:python scripts/preprocess.py
- Step 3. Train a model:python main.py --mode train --model sasrec
- Step 4. Test trained model: python main.py --mode test --model sasrec
- Step 5. Compare benchmark results: results/tables/benchmark_results.csv

---Training:

-Train Single Model:

GRU4Rec:Bashpython main.py --mode train --model gru4rec
SASRec:Bashpython main.py --mode train --model sasrec
HierSASRec:Bashpython main.py --mode train --model hier_sasrec
-Train All Models: 

python main.py --mode train --model all
Models trained:
GRU4Rec
HierGNN + GRU4Rec
SASRec
HierGNN + SASRec
NextItNet
HierGNN + NextItNet

---Testing
- Test Single Model:
python main.py --mode test --model sasrec
- Test Hierarchical Model
python main.py --mode test --model hier_nextitnet
- Test All Models:
python main.py --mode test --model all




