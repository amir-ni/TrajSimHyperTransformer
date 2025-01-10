# Multi-Level Hyper-Transformer for Trajectory Similarity

This repository provides a **complete** implementation of the _Multi-Level Hyper-Transformer_ architecture for large-scale trajectory similarity learning. The method features:

- **FourierSpatioTemporalEncoder** for advanced \((x,y,t)\) point embedding
- **Learnable Dynamic Segmentation** gating network
- **Segment-Level Mini-Transformers** for local sub-route encoding
- **Global Aggregator** for final embedding
- **Multi-Loss Synergy** (Circle, Triplet, Segment-Alignment)
- **Iterative Negative Sampling** for training efficiency
- **Train/Test** modes with checkpointing, approximate metrics, and YAML-based config

## Installation

1. Clone the repository

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Adjust configs.yaml as needed

## Running
 - Train:
   ```bash
   python src/main.py --mode train --config configs/config.yaml
   ```
 - Test:
    ```bash
   python src/main.py --mode test --config configs/config.yaml
   ```

## Configuration
All key hyperparameters and file paths are stored in configs.yaml. Update them as appropriate.

## Logging & Checkpoints
Logs are written to logs/ and checkpoints to checkpoints/ by default. Adjust paths in configs.yaml.
