# Enhanced Federated Learning Cycle for DeepFake Detection

Codespaces-first federated learning pipeline for deepfake detection using a pre-trained EfficientNet-B4 model and modular enhancement components.

## What Changed

This repository now uses a lean notebook driver that imports existing Python modules directly from the workspace instead of generating modules via `%%writefile` cells.

Primary notebook:
- `federated_learning_codespaces.ipynb`

## Current Notebook Layout

The notebook is organized into a short, practical run flow:

1. Environment and Paths (Codespaces)
- Sets deterministic seeds.
- Resolves workspace paths for frames and model file.
- Uses conservative evaluation caps to reduce memory pressure.

2. Verify Required Modules
- Checks required module files are present:
  - `enhanced_client_selection.py`
  - `update_validation.py`
  - `knowledge_distillation.py`
  - `client_reputation_ledger.py`
  - `evaluation_metrics.py`
  - `federated_learning_cycle.py`

3. Build Capped Datasets
- Builds `train_ds`, `val_ds`, `test_ds`, `proxy_ds`, `sup_ds` from frame paths.
- Applies sample caps:
  - `MAX_VAL_SAMPLES = 256`
  - `MAX_TEST_SAMPLES = 128`
  - `MAX_TRAIN_SAMPLES = 1000`
- Prints split summary so you can confirm effective sizes.

4. Configure and Start FL Pipeline
- Creates `FLCycleConfig` and initializes `FederatedLearningCycle`.
- Partitions training paths into client datasets.
- Sets up clients and enhancement components.

5. Training + Checkpoint/Resume
- Runs the federated rounds.
- Saves/reloads checkpoints from `checkpoints/`:
  - `latest_checkpoint.json`
  - `history.json`
  - `reputation_ledger.json`
  - recent `round_XXX.weights.h5`
- Prunes old weight checkpoints to limit storage pressure.
- Performs periodic lightweight/full evaluation and writes reports.

6. Export TFLite (Separate Phase)
- Export is intentionally separated from training for stability.
- Runs standard and optional quantized TFLite export after training/restore.

7. Optional Cleanup
- Includes a cleanup cell to delete `reports/` and `checkpoints/`.

## Quick Start (Codespaces)

1. Ensure these exist in workspace root:
- `ffpp_frames/` (with `real/` and `fake/` jpg frames)
- `efficientnetb4_final.keras`
- required Python modules listed above

2. Open and run notebook in order:
- `federated_learning_codespaces.ipynb`
- Run cells sequentially from top to bottom.

3. Confirm dataset sizing in the dataset cell output:
- Expected training cap is `train=1000`.

4. Start/continue training in the training cell.

5. Export TFLite in the dedicated export cell after training completes.

## Default Runtime Configuration

The current driver notebook defaults are tuned for stability in Codespaces:
- `num_devices=60`
- `clients_per_round=8`
- `local_epochs=2`
- `global_rounds=30`
- `local_batch_size=16`
- `validator_max_eval_batches=1`
- `eval_every=10`
- Distillation enabled

You can adjust these in the FL config cell based on available resources.

## Outputs

Training and evaluation artifacts are written to:
- `reports/`
- `checkpoints/`

Typical outputs include:
- round reports and comparison report JSON files
- final reputation ledger JSON
- checkpoint metadata/history/weights
- exported `.tflite` files (when export cell is run)

## Troubleshooting

### Training dataset is too large
Verify the dataset build cell shows:
- `train=1000`

If not, re-run the dataset cell before pipeline setup/training cells.

### Kernel instability near export
Use the recommended two-phase flow:
1. finish training with export disabled in training cell
2. run export in the separate export cell (ideally after a fresh kernel + checkpoint restore)

### Missing module files
Run the module verification cell and add missing files to workspace root before continuing.

### Empty training split
If the dataset cell raises an empty training assertion, reduce held-out split pressure or confirm frame discovery path is correct.

## Repository Structure (Key Files)

- `federated_learning_codespaces.ipynb` - main Codespaces notebook driver
- `federated_learning_cycle.py` - orchestration/config/training round logic
- `enhanced_client_selection.py` - client ranking and selection
- `update_validation.py` - update filtering and re-weighting
- `knowledge_distillation.py` - server-side distillation
- `client_reputation_ledger.py` - persistent reputation state
- `evaluation_metrics.py` - evaluation and report generation

## License

This project is part of a thesis workflow. Follow repository and thesis citation/usage guidance where applicable.
