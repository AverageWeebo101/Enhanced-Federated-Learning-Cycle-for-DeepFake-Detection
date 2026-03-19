# Enhanced Federated Learning Cycle for DeepFake Detection

A Jupyter Notebook implementing an **enhanced federated learning pipeline** for deepfake detection, built on top of **Flower (flwr)** with a custom TFF-compatible adapter layer. The system trains an **EfficientNet-B4** binary classifier across simulated federated clients, augmented with five thesis-specific enhancement modules.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Requirements](#requirements)
4. [Dataset](#dataset)
5. [Pre-trained Model](#pre-trained-model)
6. [Setup & Installation](#setup--installation)
7. [Notebook Walkthrough](#notebook-walkthrough)
8. [Configuration Reference](#configuration-reference)
9. [Output Files](#output-files)
10. [Quick Demo Mode](#quick-demo-mode)
11. [Resuming from Checkpoints](#resuming-from-checkpoints)
12. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a federated learning cycle where:

- **100 simulated clients** each hold a partition of video frames extracted from FaceForensics++ C23.
- A pre-trained **EfficientNet-B4** model is used as the global model.
- Each federated round applies five enhancement modules on top of standard Federated Averaging (FedAvg):
  1. **Multi-criteria Client Selection** — scores clients on validation performance, data volume, latency, reputation, and staleness.
  2. **Update Validation & Contribution Weighing** — filters and re-weights client updates based on quality.
  3. **Server-side Knowledge Distillation** — refines the global model using ensemble distillation.
  4. **Client Reputation Ledger** — maintains persistent per-client reputation scores across rounds.
  5. **Evaluation Metrics & Reporting** — generates classification reports, confusion matrices, and per-class metrics.

**Label encoding:** `0 = Real`, `1 = Fake`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Federated Learning Round                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Client Selection (Part 1)                                   │
│     └─ Multi-criteria scoring → select top-K clients            │
│  2. Flower FedAvg Round                                         │
│     └─ Broadcast model → local training → weighted averaging    │
│  3. Update Validation (Part 2)                                  │
│     └─ Contribution-weighted re-aggregation                     │
│  4. Knowledge Distillation (Part 3)                             │
│     └─ Refine global model with ensemble KD                     │
│  5. Reputation Update (Part 4)                                  │
│     └─ Update persistent client reputation ledger               │
│  6. Evaluation (Part 5)                                         │
│     └─ Periodic full evaluation with reports                    │
│  7. Inject enhanced weights back into Flower for next round     │
└─────────────────────────────────────────────────────────────────┘
```

### Module Files

The notebook writes the following Python modules to disk via `%%writefile`:

| # | Module | Description |
|---|--------|-------------|
| 1 | `enhanced_client_selection.py` | Multi-criteria client selection |
| 2 | `update_validation.py` | Update validation & contribution weighing |
| 3 | `knowledge_distillation.py` | Server-side knowledge distillation |
| 4 | `client_reputation_ledger.py` | Persistent reputation ledger |
| 5 | `evaluation_metrics.py` | Evaluation metrics & reporting |
| 6 | `federated_learning_cycle.py` | Main FL orchestrator (pure Keras) |
| 7 | `tff_data_utils.py` | Federated dataset management |
| 8 | `tff_learning_process.py` | Model wrapping & learning process |
| 9 | `tff_federated_cycle.py` | Main FL cycle orchestrator (Flower-backed) |
| 10 | `flwr_adapter.py` | Flower adapter — drop-in TFF API replacement |

> **Note:** You do **not** need to create these files manually. The notebook generates them automatically when you run the `%%writefile` cells.

---

## Requirements

### Platform

- **Kaggle Notebooks** (recommended) — GPU P100 or T4 accelerator enabled
- Google Colab (GPU runtime) is also supported
- Local Jupyter with a CUDA-capable GPU (16 GB+ VRAM recommended)

### Python Version

- Python **3.10+** (tested with 3.12)

### Python Dependencies

| Package | Min Version | Purpose |
|---------|-------------|---------|
| `tensorflow` | 2.17+ | Deep learning framework, model training |
| `flwr` (Flower) | 1.7+ | Federated learning backend |
| `numpy` | 1.24+ | Numerical operations |
| `opencv-python` (`cv2`) | 4.8+ | Video frame extraction |
| `matplotlib` | 3.7+ | Results visualization |
| `scikit-learn` | 1.3+ | Evaluation metrics (classification report, confusion matrix) |
| `tqdm` | 4.65+ | Progress bars |

> **TFF Note:** TensorFlow Federated (TFF 0.86.0) is **not required**. The notebook uses a custom Flower-based adapter (`flwr_adapter.py`) that reimplements the TFF API surface used by this project. This avoids TFF's incompatibility with Python ≥ 3.12 and TF ≥ 2.17.

### Hardware

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 16 GB | 16 GB (P100/T4) |
| System RAM | 16 GB | 29 GB (Kaggle default) |
| Disk Space | 20 GB | 57+ GB |

The notebook supports **TPU**, **GPU**, and **CPU** runtimes, with automatic detection and mixed-precision configuration:
- **TPU** → `mixed_bfloat16`
- **GPU** → `mixed_float16` + XLA JIT
- **CPU** → no mixed precision (significantly slower)

---

## Dataset

### FaceForensics++ C23

The notebook uses the **FaceForensics++ C23** dataset, available on Kaggle.

**Kaggle dataset slug:** `xdxd003/ff-c23`

#### Adding the Dataset on Kaggle

1. Open the notebook on Kaggle.
2. In the right sidebar, click **Add data** → **Search datasets**.
3. Search for `xdxd003/ff-c23` and add it.
4. The dataset will be auto-mounted at `/kaggle/input/` — **no manual download required**.

#### Expected Directory Structure

The notebook expects the following structure under the Kaggle mount:

```
/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23/
├── original/              # Real videos (.mp4)
├── Deepfakes/             # Fake videos — Deepfakes method
├── FaceSwap/              # Fake videos — FaceSwap method
├── Face2Face/             # (available but commented out by default)
└── NeuralTextures/        # (available but commented out by default)
```

#### Fake Methods Used (Default)

By default, the notebook uses **Deepfakes** and **FaceSwap** manipulation methods. Face2Face and NeuralTextures are available but commented out in the extraction cell. To enable them, uncomment the corresponding lines in the `FAKE_DIRS` list in Cell 42.

#### Frame Extraction Details

- **Adaptive sampling:** 8–20 frames per video depending on video length
- **Resolution:** 260×260 pixels (resized during extraction)
- **Format:** JPEG (quality 90)
- **Parallelism:** Up to 8 workers (auto-capped to CPU count)
- Frames are saved to `/kaggle/working/ffpp_frames/real/` and `/kaggle/working/ffpp_frames/fake/`

#### Data Splits

After extraction, frames are shuffled (seed=42) and split into:

| Split | Percentage | Purpose |
|-------|-----------|---------|
| Validation (`val_ds`) | 15% | Server-side model validation during training |
| Test (`test_ds`) | 10% | Final held-out evaluation |
| Proxy (`proxy_ds`) | 1.5% | Unlabelled data for knowledge distillation |
| Supervised (`sup_ds`) | 2% | Labelled data for supervised distillation loss |
| Client Shards | Remaining | Partitioned across 100 federated clients as TFRecords |

---

## Pre-trained Model

The notebook requires a pre-trained **EfficientNet-B4** binary classifier saved as `efficientnetb4_final.keras`.

### How to Provide the Model

**Option A — Kaggle Input Dataset (recommended):**
Upload the `.keras` file as a Kaggle dataset and add it as an input to your notebook. The notebook will auto-detect it under `/kaggle/input/`.

**Option B — Direct Upload:**
Place `efficientnetb4_final.keras` in the working directory (`/kaggle/working/`) before running the notebook.

**Option C — Google Colab Upload:**
Uncomment the upload cell (Cell 28) and follow the browser upload prompt.

The notebook searches the following paths automatically:
- `/kaggle/input/*/efficientnetb4_final.keras`
- `/kaggle/working/efficientnetb4_final.keras`
- Google Drive mount paths

---

## Setup & Installation

### On Kaggle (Recommended)

1. **Create a new notebook** on Kaggle or import this `.ipynb` file.
2. **Enable GPU accelerator:** Settings → Accelerator → GPU P100 or T4.
3. **Add the dataset:** Add data → search `xdxd003/ff-c23` → Add.
4. **Add the pre-trained model:** Add your `efficientnetb4_final.keras` as an input dataset.
5. **Run all cells sequentially** from top to bottom.

### On Google Colab

1. Upload the notebook to Colab.
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Upload `efficientnetb4_final.keras` when prompted, or mount Google Drive.
4. Download the FF++ C23 dataset manually and adjust `DATA_DIR` paths.
5. Run all cells sequentially.

### Local Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow>=2.17 flwr>=1.7 numpy opencv-python matplotlib scikit-learn tqdm
   ```

3. Launch Jupyter:
   ```bash
   pip install jupyter
   jupyter notebook enhancedfederatedlearningcyclefordeepfakedetection.ipynb
   ```

4. Place `efficientnetb4_final.keras` in the same directory as the notebook.

5. Download the FaceForensics++ C23 dataset and update `DATA_DIR` in the frame extraction cell to point to its location.

---

## Notebook Walkthrough

Run the cells **sequentially from top to bottom**. Below is a summary of each section:

### Section 1 — Environment Setup (Cells 1–7)

- Installs Flower (`flwr>=1.7`) via pip.
- Writes the `flwr_adapter.py` module to disk (provides TFF-compatible API).
- Registers a custom Keras `preprocess_input` function for serialization.
- Verifies environment: TensorFlow version, GPU/TPU availability, adapter imports.
- Configures accelerator (TPU/GPU/CPU), mixed precision, and XLA JIT.

### Section 2 — Write Module Files (Cells 8–26)

- Uses `%%writefile` to create 10 Python modules in the working directory.
- **Run all cells in this section**—they produce the `.py` files that the rest of the notebook imports.
- Modules cover Parts 1–5 (enhancements), the main FL orchestrator, data utilities, learning process, and the Flower-backed cycle.

### Section 3 — Load Pre-trained Model (Cells 27–30)

- Locates `efficientnetb4_final.keras` across candidate paths (Kaggle input, working directory, Google Drive).
- Copies the model to the working directory if found elsewhere.
- Verifies file size and hash.

### Section 4 — Quick Verification (Cell 31)

- Imports all modules and checks for import errors.
- Loads the model and prints a summary with parameter count and input shape.

### Section 5 — Configuration (Cell 32)

- Creates a `TFFCycleConfig` object with all hyperparameters.
- This is the **single place to adjust all experiment settings** (see [Configuration Reference](#configuration-reference)).

### Section 6 — Data Preparation (Cells 33–44)

- Locates the FF++ C23 dataset under `/kaggle/input/`.
- Extracts video frames using OpenCV (parallel, adaptive sampling).
- Partitions frames across 100 clients as GZIP-compressed TFRecord shards.
- Builds `val_ds`, `test_ds`, `proxy_ds`, and `sup_ds`.

### Section 7 — Initialize Flower Process (Cells 45–49)

- Defines the TFRecord parser for client shards.
- Reloads all `%%writefile` modules to pick up any edits.
- Instantiates `TFFFederatedLearningCycle`, loads the global model, creates federated clients.
- Sets up Flower process and enhancement modules (Parts 1–5).

### Section 8 — Run Federated Learning Cycle (Cells 50–58)

- Defines checkpoint and memory management utilities.
- Builds validation, test, proxy, and supervised datasets from extracted frames.
- Runs the full federated training loop with automatic checkpointing.
- Each round: Client Selection → FedAvg → Update Validation → Knowledge Distillation → Reputation Update → Evaluation.

### Section 9 — Results Visualization (Cells 59–64)

- Plots accuracy and loss curves across rounds.
- Visualizes reputation distribution.
- Recovers history from checkpoints if the variable is missing.

### Section 10 — Final Evaluation Report (Cells 65–66)

- Runs `evaluate_and_report()` on the held-out test set.
- Generates classification metrics (precision, recall, F1), confusion matrix, and summary statistics.

### Section 11 — Export & Download (Cells 67–69)

- Saves the trained model as `.h5`.
- Exports TF Lite models (standard + quantised).
- Bundles all outputs into `fl_cycle_results.zip`.
- On Colab, triggers a browser download. On Kaggle, outputs appear in the Output tab.

### Appendix A — Quick Demo (Cell 70)

- Commented-out quick test with 8 clients and 3 rounds (see [Quick Demo Mode](#quick-demo-mode)).

---

## Configuration Reference

All hyperparameters are set in the `TFFCycleConfig` object (Section 5, Cell 32):

### Core FL Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `'efficientnetb4_final.keras'` | Path to the pre-trained Keras model |
| `num_devices` | `100` | Total number of simulated federated clients |
| `local_epochs` | `5` | Local training epochs per client per round |
| `global_rounds` | `50` | Total federated aggregation rounds |
| `clients_per_round` | `15` | Clients selected per round |
| `local_batch_size` | `32` | Batch size for client-side training |
| `local_lr` | `1e-4` | Client-side learning rate |
| `server_lr` | `0.1` | Server-side learning rate (FedAvg scale) |
| `eval_every` | `10` | Full evaluation every N rounds |
| `client_optimizer` | `'adam'` | Client-side optimizer |
| `server_optimizer` | `'sgd'` | Server-side optimizer |
| `enable_comparison` | `False` | Log both baseline FedAvg and enhanced accuracy (uses more RAM) |

### Knowledge Distillation (Part 3)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_distillation` | `True` | Enable server-side knowledge distillation |
| `temperature` | `2.0` | Softmax temperature for KD |
| `lam` | `0.5` | Balance between soft and hard targets |
| `epochs` | `3` | Distillation fine-tuning epochs |
| `batch_size` | `32` | Distillation batch size |
| `learning_rate` | `1e-4` | Distillation learning rate |

### Client Selection Weights (Part 1)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_v` | `0.30` | Weight for local validation performance |
| `w_d` | `0.20` | Weight for data volume |
| `w_l` | `0.10` | Weight for latency (applied to 1 − L_i) |
| `w_r` | `0.25` | Weight for reputation |
| `w_s` | `0.15` | Weight for staleness penalty |

### Contribution Weights (Part 2)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.35` | Validation gain weight |
| `beta` | `0.20` | Similarity to global update history |
| `gamma` | `0.20` | Data volume weight |
| `delta` | `0.25` | Reputation weight |
| `clip_threshold` | `10.0` | Gradient clipping threshold |
| `clip_value` | `5.0` | Gradient clipping value |
| `harmful_threshold` | `0.02` | Threshold for identifying harmful updates |

### Reputation Ledger (Part 4)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta` | `0.0` | Reputation gain threshold |
| `gamma` | `0.10` | Reputation learning rate |
| `decay_rate` | `0.99` | Per-round reputation decay |
| `floor` | `0.05` | Minimum reputation score |
| `ceiling` | `1.0` | Maximum reputation score |
| `initial_reputation` | `0.50` | Starting reputation for new clients |
| `penalty_factor` | `0.05` | Penalty for poor contributions |

---

## Output Files

After a complete run, the following files are generated:

| File | Description |
|------|-------------|
| `effnet_global_tff_trained.h5` | Final trained global model (Keras HDF5) |
| `effnet_global_tff_final.tflite` | TF Lite model (standard) |
| `effnet_global_tff_final_quantised.tflite` | TF Lite model (quantised) |
| `reports/` | Evaluation reports (JSON, classification metrics) |
| `results_overview.png` | Accuracy and loss curves |
| `reputation_distribution.png` | Client reputation distribution |
| `fl_cycle_results.zip` | ZIP archive of all outputs |
| `fl_checkpoints/` | Round checkpoints (model weights, history, reputation) |

On **Kaggle**, all files in `/kaggle/working/` appear in the notebook's **Output** tab.

---

## Quick Demo Mode

For a faster smoke-test, uncomment the code in the **Appendix A** cell (last code cell) and comment out Section 8. This runs:

- **8 clients** instead of 100
- **3 rounds** instead of 50
- **Synthetic data** (no dataset download required)
- Comparison mode enabled

This is useful for verifying the pipeline works before committing to a full run.

---

## Resuming from Checkpoints

The notebook automatically saves checkpoints during training. If your session is interrupted (e.g., Kaggle timeout):

1. **Re-run Sections 1–7** to rebuild the environment, modules, and cycle object.
2. **Run the Session Recovery cell** (Cell 57 in Section 8) — it rebuilds the full runtime state from saved checkpoint files.
3. **Re-run the training cell** — it will automatically resume from the last saved checkpoint.

Checkpoints include:
- Model weights (`.weights.h5`)
- Training history (JSON)
- Reputation ledger state
- Current round number

---

## Troubleshooting

### `flwr_adapter.py` import fails with SyntaxError
The notebook includes built-in diagnostics. Check the cell output for unterminated triple-quote strings (odd count of `"""` or `'''`). Re-run the `%%writefile` cell for `flwr_adapter.py`.

### `preprocess_input` deserialization error
Make sure you run the cell that registers `preprocess_input` with `@register_keras_serializable` (Cell 2) **before** loading or cloning the model.

### Out of Memory (OOM) on Kaggle
- Reduce `clients_per_round` (e.g., from 15 to 8).
- Reduce `local_batch_size` (e.g., from 32 to 16).
- Set `enable_comparison=False` (avoids running a second baseline FedAvg).
- Reduce `num_devices` for initial testing.

### No GPU detected
- On Kaggle: Settings → Accelerator → GPU P100 or T4.
- On Colab: Runtime → Change runtime type → T4 GPU.
- Locally: Ensure CUDA and cuDNN are installed and `tensorflow-gpu` is properly configured.

### No videos found during frame extraction
- Verify the dataset is added as an input on Kaggle.
- Check `DATA_DIR` points to the correct path. The notebook auto-detects both old-style (`/kaggle/input/ff-c23/`) and new-style (`/kaggle/input/datasets/xdxd003/ff-c23/`) Kaggle mount layouts.

### TFRecord files missing after kernel restart
Re-run the TFRecord creation cell (Cell 44). Alternatively, if frames already exist on disk, the extraction cell will skip re-extraction and you only need to rebuild the TFRecords.

---

## License

This project is part of a thesis. Please refer to the thesis document for usage and citation guidelines.
