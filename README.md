# Malaysian Dialect Translation

Fine-tuning T5 models for translation between Malaysian dialects and standard languages (Malay, English, Manglish).

## Overview

This project fine-tunes T5 models for translation tasks. We trained two models:

- **Small Model** (`mesolitica/nanot5-small-malaysian-cased`): Two-stage training approach
  - **Stage 1**: Standard language pairs (Melayu ↔ Inggeris)
  - **Stage 2**: Dialect translation pairs (10 Malaysian dialects)
- **Base Model**: Trained directly on combined data from both stages (non-staged training)

## Features

- Data processing pipeline with cleaning and dialect detection
- Two-stage training approach
- Single GPU and multi-GPU (DDP) training support
- Weights & Biases integration for experiment tracking

## Setup

### Prerequisites

- Python >= 3.12
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies:
```bash
uv sync
```

Or with pip:
```bash
pip install -e .
```

3. (Optional) Install dev dependencies for notebooks:
```bash
uv sync --group dev
```

## Reproduction Steps

### 1. Data Processing

Process and prepare the dataset:

```bash
python src/processor.py
```

This downloads data from `mesolitica/Malaysian-Translation`, cleans it, adds dialect detection, and creates train/val/test splits for both stages. Outputs are saved to `src/data/processed/dataset/stage1/` and `src/data/processed/dataset/stage2/`.

### 2. Training

#### Small Model (Two-Stage Training)

**Stage 1 Training (Single GPU)**
```bash
python src/trainer.py configs/train_config.yaml
```

**Stage 2 Training (Single GPU)**
```bash
python src/trainer.py configs/train_config_stage2.yaml
```

#### Base Model (Non-Staged Training)

The base model was trained directly on the combined dataset (all data from stage 1 and stage 2 together) without the two-stage approach. To reproduce this, you can combine the stage 1 and stage 2 datasets or modify the processor to save the combined dataset before stage splitting.

#### Multi-GPU Training

```bash
python src/trainer_ddp.py configs/train_config.yaml
```

Or with `torchrun`:
```bash
torchrun --nproc_per_node=<num_gpus> src/trainer_ddp.py configs/train_config.yaml
```

### 3. Evaluation

Use the provided notebooks:
```bash
jupyter notebook notebooks/model_eval.ipynb
```

## Dataset Derivation

The dataset is derived from `mesolitica/Malaysian-Translation` on HuggingFace through the following pipeline:

### 1. Data Filtering
- Filters by allowed translation prefixes (standard languages and 10 Malaysian dialects)
- Removes self-translations (where source equals target)
- Removes empty or excessively long sequences (>1000 characters)
- Removes code blocks and text with too many non-alphanumeric characters
- Removes entries starting with code block markers

### 2. Dialect Detection
- Adds heuristic-based dialect detection from source text using word matching.Keywods are mined from the dataset based on targets ([`src/dialect_detector.py`](src/dialect_detector.py), [`src/processor.py`](src/processor.py#L76-L96))
- Creates `direction_pair` labels (e.g., "Melayu_Inggeris", "Kelantan_Sabah")
- Filters out entries with unknown detected dialects

### 3. Stage Splitting
- **Stage 1**: First 200k examples of standard language pairs (Melayu ↔ Inggeris)
- **Stage 2**: All dialect translation pairs + remaining standard language pairs

### 4. Train/Val/Test Splits
- Creates semantic keys from source text (normalized and hashed) for deduplication
- Splits by unique semantic keys (80/10/10) to prevent data leakage
- Ensures same source text doesn't appear in multiple splits

## Configuration

Training is configured via YAML files in `configs/`. Edit the config files to adjust:
- Data paths
- Model parameters
- Training hyperparameters
- Output directory
- W&B project settings

## Caveats

- **Dialect Detection**: The dialect detection is heuristic-based using word matching. Keywords are mined from the dataset based on target labels, which may not generalize perfectly to new text. Unknown dialects are filtered out, potentially reducing dataset size.
- **Data Quality**: The filtering pipeline removes code blocks and non-alphanumeric-heavy text, which may exclude some valid translation pairs that contain technical terms or formatting.
- **Stage Splitting**: Stage 1 uses only the first 200k standard language pairs, which may not represent the full diversity of the available data. The remaining standard pairs are combined with dialect pairs in Stage 2.
- **Semantic Deduplication**: Train/val/test splits are based on semantic keys (hashed source text), which prevents exact duplicates but may still allow semantically similar examples across splits.

## Acknowledgments

- Base Model: `mesolitica/nanot5-small-malaysian-cased`
- Small Model: Fine-tuned from `mesolitica/nanot5-small-malaysian-cased`
- Dataset: `mesolitica/Malaysian-Translation`
