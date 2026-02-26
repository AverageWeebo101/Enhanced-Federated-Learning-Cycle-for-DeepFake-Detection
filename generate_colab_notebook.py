"""
Generate a self-contained Google Colab notebook (.ipynb) that embeds
all 9 module files via %%writefile cells, installs TFF, uploads the
model, runs the full Enhanced Federated Learning Cycle, and visualises
results.

Run:
    python generate_colab_notebook.py

Output:
    Enhanced_FL_Cycle_DeepFake_Detection.ipynb
"""

import json
from pathlib import Path

# ── Ordered list of module files to embed ──────────────────────────────
MODULE_FILES = [
    "enhanced_client_selection.py",
    "update_validation.py",
    "knowledge_distillation.py",
    "client_reputation_ledger.py",
    "evaluation_metrics.py",
    "federated_learning_cycle.py",
    "tff_data_utils.py",
    "tff_learning_process.py",
    "tff_federated_cycle.py",
]

PART_DESCRIPTIONS = {
    "enhanced_client_selection.py": "Part 1 — Enhanced Client Selection (multi-criteria scoring)",
    "update_validation.py": "Part 2 — Update Validation & Contribution Weighing",
    "knowledge_distillation.py": "Part 3 — Server-Side Knowledge Distillation",
    "client_reputation_ledger.py": "Part 4 — Client Reputation & Persistent Ledger",
    "evaluation_metrics.py": "Part 5 — Evaluation Metrics & Report Generation",
    "federated_learning_cycle.py": "Part 6 — Main FL Cycle Orchestrator (non-TFF, pure Keras)",
    "tff_data_utils.py": "TFF Wrapper — Federated Dataset Management",
    "tff_learning_process.py": "TFF Wrapper — Model Wrapping & Learning Process",
    "tff_federated_cycle.py": "TFF Wrapper — Main TFF-Based Orchestrator",
}

# ── Helper to build cells ──────────────────────────────────────────────

def md_cell(source: str) -> dict:
    """Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }

def code_cell(source: str) -> dict:
    """Code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "execution_count": None,
        "outputs": [],
    }


# ── Build the notebook ─────────────────────────────────────────────────

def build_notebook() -> dict:
    cells = []

    # ════════════════════════════════════════════════════════════════════
    #  Title & overview
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "# Enhanced Federated Learning Cycle for DeepFake Detection\n"
        "\n"
        "**Thesis Project — Google Colab Runtime**\n"
        "\n"
        "This notebook is **self-contained**: it installs all dependencies,\n"
        "writes all 9 Python module files, uploads the pre-trained model,\n"
        "and runs the full Enhanced Federated Learning Cycle using\n"
        "**TensorFlow Federated (TFF)**.\n"
        "\n"
        "## Modules\n"
        "\n"
        "| # | Module | Description |\n"
        "|---|--------|-------------|\n"
        "| 1 | `enhanced_client_selection.py` | Multi-criteria client selection |\n"
        "| 2 | `update_validation.py` | Update validation & contribution weighing |\n"
        "| 3 | `knowledge_distillation.py` | Server-side knowledge distillation |\n"
        "| 4 | `client_reputation_ledger.py` | Persistent reputation ledger |\n"
        "| 5 | `evaluation_metrics.py` | Evaluation metrics & reporting |\n"
        "| 6 | `federated_learning_cycle.py` | Main FL orchestrator (pure Keras) |\n"
        "| 7 | `tff_data_utils.py` | TFF data management wrapper |\n"
        "| 8 | `tff_learning_process.py` | TFF model wrapping & process builder |\n"
        "| 9 | `tff_federated_cycle.py` | TFF-based FL cycle orchestrator |\n"
        "\n"
        "## Architecture\n"
        "\n"
        "TFF handles the **core federated computation** (model broadcast →\n"
        "client-side local training → data-weighted FedAvg). Our thesis\n"
        "enhancements operate as a **post-aggregation refinement layer**:\n"
        "\n"
        "1. **Client Selection** (Part 1) → selects participants\n"
        "2. **TFF Round** → FedAvg on selected clients\n"
        "3. **Update Validation** (Part 2) → contribution-weighted re-aggregation\n"
        "4. **Knowledge Distillation** (Part 3) → refine with ensemble KD\n"
        "5. **Reputation Update** (Part 4) → feed gains into ledger\n"
        "6. **Evaluation** (Part 5) → periodic full eval with reports\n"
        "7. Inject enhanced weights back into TFF state for next round\n"
        "\n"
        "---\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 1: Environment Setup
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 1. Environment Setup\n"
        "\n"
        "Install the compatible TensorFlow + TFF stack.\n"
        "\n"
        "> **Important:** After running the install cell, you may need to\n"
        "> **restart the runtime** (Runtime → Restart runtime) before\n"
        "> proceeding. Colab will prompt you if needed.\n"
    ))

    cells.append(code_cell(
        "# ── Install compatible TF + TFF stack ──────────────────────────\n"
        "# TFF 0.86.0 is the latest version compatible with TF 2.16.x\n"
        "# (which Colab typically provides). Adjust if your Colab has\n"
        "# a different TF version.\n"
        "#\n"
        "# Option A: Use Colab's pre-installed TF + matching TFF\n"
        "import tensorflow as tf\n"
        "print(f'Current TF version: {tf.__version__}')\n"
        "\n"
        "# Install TFF matching the current TF version\n"
        "# See: https://www.tensorflow.org/federated/install\n"
        "!pip install -q tensorflow-federated\n"
        "\n"
        "# If the above fails, try pinning specific versions:\n"
        "# !pip install -q tensorflow==2.14.1 tensorflow-federated==0.72.0\n"
    ))

    cells.append(code_cell(
        "# ── Verify installation ────────────────────────────────────────\n"
        "import tensorflow as tf\n"
        "import tensorflow_federated as tff\n"
        "import numpy as np\n"
        "\n"
        "print(f'TensorFlow  : {tf.__version__}')\n"
        "print(f'TFF         : {tff.__version__}')\n"
        "print(f'NumPy       : {np.__version__}')\n"
        "print(f'GPU         : {tf.config.list_physical_devices(\"GPU\")}')\n"
        "print('\\n✅ Environment ready.')\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 2: Write Module Files
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 2. Write Module Files\n"
        "\n"
        "Each cell below uses `%%writefile` to create the corresponding\n"
        "Python module in the Colab working directory. Run them all sequentially.\n"
    ))

    for fname in MODULE_FILES:
        desc = PART_DESCRIPTIONS.get(fname, fname)
        fpath = Path(__file__).parent / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found, skipping.")
            continue

        content = fpath.read_text(encoding="utf-8")

        cells.append(md_cell(f"### {desc}\n"))
        cells.append(code_cell(f"%%writefile {fname}\n{content}"))

    # ════════════════════════════════════════════════════════════════════
    #  Section 3: Upload Model
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 3. Upload Pre-trained Model\n"
        "\n"
        "Upload `effnet_ffpp_small_data.h5` (the EfficientNet binary classifier\n"
        "for DeepFake detection). Choose **one** of the methods below.\n"
        "\n"
        "### Option A: Upload from local machine\n"
    ))

    cells.append(code_cell(
        "# ── Option A: Upload from your local machine ──────────────────\n"
        "from google.colab import files\n"
        "import os\n"
        "\n"
        "if not os.path.exists('effnet_ffpp_small_data.h5'):\n"
        "    print('Please upload effnet_ffpp_small_data.h5')\n"
        "    uploaded = files.upload()\n"
        "    print(f'Uploaded: {list(uploaded.keys())}')\n"
        "else:\n"
        "    print('Model file already present.')\n"
        "\n"
        "# Verify\n"
        "size_mb = os.path.getsize('effnet_ffpp_small_data.h5') / (1024**2)\n"
        "print(f'Model file: effnet_ffpp_small_data.h5 ({size_mb:.1f} MB)')\n"
    ))

    cells.append(md_cell(
        "### Option B: Mount Google Drive\n"
        "\n"
        "If the model is stored in your Google Drive, mount it and copy.\n"
    ))

    cells.append(code_cell(
        "# ── Option B: Mount Google Drive ──────────────────────────────\n"
        "# from google.colab import drive\n"
        "# drive.mount('/content/drive')\n"
        "#\n"
        "# # Adjust the path to where your model is stored in Drive:\n"
        "# DRIVE_MODEL_PATH = '/content/drive/MyDrive/thesis/effnet_ffpp_small_data.h5'\n"
        "#\n"
        "# import shutil\n"
        "# shutil.copy(DRIVE_MODEL_PATH, 'effnet_ffpp_small_data.h5')\n"
        "# print('Model copied from Google Drive.')\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 4: Quick Verification
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 4. Quick Verification\n"
        "\n"
        "Verify that all modules import correctly and the model loads.\n"
    ))

    cells.append(code_cell(
        "# ── Verify module imports ─────────────────────────────────────\n"
        "import enhanced_client_selection\n"
        "import update_validation\n"
        "import knowledge_distillation\n"
        "import client_reputation_ledger\n"
        "import evaluation_metrics\n"
        "import federated_learning_cycle\n"
        "import tff_data_utils\n"
        "import tff_learning_process\n"
        "import tff_federated_cycle\n"
        "\n"
        "print('✅ All 9 modules imported successfully.')\n"
        "\n"
        "# Verify TFF is detected\n"
        "from tff_data_utils import TFF_AVAILABLE\n"
        "print(f'TFF Available: {TFF_AVAILABLE}')\n"
        "assert TFF_AVAILABLE, 'TFF must be available for the full cycle!'\n"
        "\n"
        "# Load and verify model\n"
        "import tensorflow as tf\n"
        "model = tf.keras.models.load_model('effnet_ffpp_small_data.h5', compile=False)\n"
        "print(f'\\nModel loaded: {model.count_params():,} params')\n"
        "print(f'Input shape:  {model.input_shape}')\n"
        "print(f'Output shape: {model.output_shape}')\n"
        "model.summary()\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 5: Configuration
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 5. Configuration\n"
        "\n"
        "Configure the federated learning cycle parameters.\n"
        "Adjust these values based on your experiment requirements.\n"
    ))

    cells.append(code_cell(
        "# ── Experiment Configuration ──────────────────────────────────\n"
        "from tff_federated_cycle import TFFCycleConfig\n"
        "from knowledge_distillation import DistillationConfig\n"
        "from enhanced_client_selection import SelectionWeights\n"
        "from update_validation import ContributionWeights, ClippingConfig\n"
        "from client_reputation_ledger import ReputationConfig\n"
        "\n"
        "config = TFFCycleConfig(\n"
        "    # ── Core FL settings ────────────────────────────────────\n"
        "    model_path='effnet_ffpp_small_data.h5',\n"
        "    num_devices=100,           # Total number of simulated clients\n"
        "    local_epochs=5,            # Local training epochs per client per round\n"
        "    global_rounds=50,          # Total federated aggregation rounds\n"
        "    clients_per_round=15,      # Clients selected each round\n"
        "    local_batch_size=32,\n"
        "    local_lr=1e-4,             # Client-side learning rate\n"
        "    server_lr=1.0,             # Server-side learning rate (FedAvg scale)\n"
        "    eval_every=10,             # Full evaluation every N rounds\n"
        "\n"
        "    # ── TFF process settings ────────────────────────────────\n"
        "    client_optimizer='adam',\n"
        "    server_optimizer='sgd',\n"
        "    enable_comparison=True,    # Log TFF FedAvg vs enhanced accuracy\n"
        "\n"
        "    # ── Knowledge Distillation (Part 3) ─────────────────────\n"
        "    enable_distillation=True,\n"
        "    distillation_config=DistillationConfig(\n"
        "        temperature=3.0,\n"
        "        lam=0.7,\n"
        "        epochs=3,\n"
        "        batch_size=32,\n"
        "        learning_rate=1e-4,\n"
        "    ),\n"
        "\n"
        "    # ── Client Selection Weights (Part 1) ───────────────────\n"
        "    selection_weights=SelectionWeights(\n"
        "        w_v=0.30,   # Local validation performance\n"
        "        w_d=0.20,   # Data volume\n"
        "        w_l=0.10,   # Latency (applied to 1 - L_i)\n"
        "        w_r=0.25,   # Reputation\n"
        "        w_s=0.15,   # Staleness penalty\n"
        "    ),\n"
        "\n"
        "    # ── Contribution Weights (Part 2) ───────────────────────\n"
        "    contribution_weights=ContributionWeights(\n"
        "        alpha=0.35,   # Validation gain\n"
        "        beta=0.20,    # Similarity to global update history\n"
        "        gamma=0.20,   # Data volume\n"
        "        delta=0.25,   # Reputation\n"
        "    ),\n"
        "    clipping_config=ClippingConfig(\n"
        "        clip_threshold=10.0,\n"
        "        clip_value=5.0,\n"
        "    ),\n"
        "    harmful_threshold=0.02,\n"
        "\n"
        "    # ── Reputation Ledger (Part 4) ──────────────────────────\n"
        "    reputation_config=ReputationConfig(\n"
        "        theta=0.0,\n"
        "        gamma=0.10,\n"
        "        decay_rate=0.99,\n"
        "        floor=0.05,\n"
        "        ceiling=1.0,\n"
        "        initial_reputation=0.50,\n"
        "        penalty_factor=0.05,\n"
        "    ),\n"
        "\n"
        "    # ── Output ──────────────────────────────────────────────\n"
        "    reports_dir='reports',\n"
        "    tflite_output_path='effnet_global_tff_final.tflite',\n"
        ")\n"
        "\n"
        "print('Configuration created.')\n"
        "print(f'  Devices:         {config.num_devices}')\n"
        "print(f'  Rounds:          {config.global_rounds}')\n"
        "print(f'  Local epochs:    {config.local_epochs}')\n"
        "print(f'  Clients/round:   {config.clients_per_round}')\n"
        "print(f'  Distillation:    {config.enable_distillation}')\n"
        "print(f'  Comparison mode: {config.enable_comparison}')\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 6: Data Preparation
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 6. Data Preparation\n"
        "\n"
        "Prepare the training, validation, test, and proxy datasets.\n"
        "\n"
        "> **Note:** This uses **synthetic data** for demonstration.\n"
        "> Replace the generators below with your real FF++ c23 data\n"
        "> loaders for actual experiments:\n"
        "> ```python\n"
        "> train_data = load_ffpp_c23_train(...)      # for client partitions\n"
        "> val_data   = load_ffpp_c23_val(...)        # server validation\n"
        "> test_data  = load_ffpp_c23_test(...)       # independent test set\n"
        "> proxy_data = load_ffpp_c23_unlabelled(...) # for distillation\n"
        "> ```\n"
    ))

    cells.append(code_cell(
        "import numpy as np\n"
        "import tensorflow as tf\n"
        "from tff_data_utils import (\n"
        "    partition_data_iid_tff,\n"
        "    generate_synthetic_data,\n"
        "    generate_proxy_data,\n"
        ")\n"
        "from tff_federated_cycle import TFFFederatedLearningCycle\n"
        "\n"
        "np.random.seed(42)\n"
        "tf.random.set_seed(42)\n"
        "\n"
        "# ── Load model and get input shape ────────────────────────────\n"
        "cycle = TFFFederatedLearningCycle(config)\n"
        "model = cycle.load_global_model()\n"
        "input_shape = model.input_shape[1:]  # Strip batch dim\n"
        "config.input_shape = input_shape\n"
        "print(f'Model input shape: {input_shape}')\n"
        "\n"
        "# ── Generate synthetic datasets ───────────────────────────────\n"
        "# Replace these with real FF++ c23 data loaders\n"
        "TOTAL_TRAIN = config.num_devices * 10   # 10 samples per client\n"
        "VAL_SAMPLES   = 200\n"
        "TEST_SAMPLES  = 300\n"
        "PROXY_SAMPLES = 150\n"
        "SUP_SAMPLES   = 100\n"
        "\n"
        "print(f'Generating synthetic data ...')\n"
        "print(f'  Training:    {TOTAL_TRAIN} samples (partitioned across {config.num_devices} clients)')\n"
        "print(f'  Validation:  {VAL_SAMPLES} samples')\n"
        "print(f'  Test:        {TEST_SAMPLES} samples')\n"
        "print(f'  Proxy (KD):  {PROXY_SAMPLES} samples')\n"
        "print(f'  Supervised:  {SUP_SAMPLES} samples')\n"
        "\n"
        "train_ds  = generate_synthetic_data(TOTAL_TRAIN, input_shape, seed=1)\n"
        "val_ds    = generate_synthetic_data(VAL_SAMPLES,  input_shape, seed=2)\n"
        "test_ds   = generate_synthetic_data(TEST_SAMPLES, input_shape, seed=3)\n"
        "proxy_ds  = generate_proxy_data(PROXY_SAMPLES,    input_shape, seed=4)\n"
        "sup_ds    = generate_synthetic_data(SUP_SAMPLES,   input_shape, seed=5)\n"
        "\n"
        "# ── IID Partition across clients ──────────────────────────────\n"
        "client_data = partition_data_iid_tff(train_ds, config.num_devices, seed=42)\n"
        "print(f'\\nPartitioned into {len(client_data)} client shards.')\n"
        "\n"
        "# ── Create FL clients ─────────────────────────────────────────\n"
        "cycle.create_clients(client_data)\n"
        "print(f'Created {len(cycle.clients)} federated clients.')\n"
        "print('\\n✅ Data preparation complete.')\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 7: Initialize TFF Process & Enhancement Modules
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 7. Initialize TFF Process & Enhancement Modules\n"
        "\n"
        "Build the TFF Weighted FedAvg learning process and wire up\n"
        "all five enhancement modules.\n"
    ))

    cells.append(code_cell(
        "# ── Setup TFF federated learning process ─────────────────────\n"
        "cycle.setup_tff_process()\n"
        "print('TFF Learning Process initialised.')\n"
        "\n"
        "# ── Setup enhancement modules (Parts 1-5) ────────────────────\n"
        "cycle.setup_enhancement_modules()\n"
        "print('Enhancement modules (Parts 1-5) initialised.')\n"
        "print('\\n✅ Ready to run the federated learning cycle.')\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 8: Run the Full TFF Federated Learning Cycle
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 8. Run the Full TFF Federated Learning Cycle\n"
        "\n"
        "Execute the complete federated training loop:\n"
        "- Each round: Client Selection → TFF FedAvg → Update Validation →\n"
        "  Knowledge Distillation → Reputation Update → Evaluation\n"
        "- Comparison mode logs both TFF FedAvg and enhanced accuracy\n"
        "- Full evaluation reports saved every `eval_every` rounds\n"
        "\n"
        "> **Runtime:** With 100 clients, 50 rounds, and EfficientNet,\n"
        "> this may take significant time. Consider reducing\n"
        "> `global_rounds` or `num_devices` for initial testing.\n"
    ))

    cells.append(code_cell(
        "# ── Run the full federated learning cycle ────────────────────\n"
        "history = cycle.run(\n"
        "    server_val_data=val_ds,\n"
        "    test_data=test_ds,\n"
        "    proxy_data=proxy_ds,\n"
        "    supervised_data=sup_ds,\n"
        ")\n"
        "\n"
        "print('\\n✅ Federated Learning Cycle complete!')\n"
        "print(f'History keys: {list(history.keys())}')\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 9: Results Visualization
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 9. Results Visualization\n"
        "\n"
        "Visualise the training history with matplotlib.\n"
    ))

    cells.append(code_cell(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        "# ── 9a. Accuracy: TFF FedAvg vs Enhanced ────────────────────\n"
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n"
        "\n"
        "rounds = history['round']\n"
        "tff_acc = history['tff_fedavg_accuracy']\n"
        "enh_acc = history['enhanced_accuracy']\n"
        "\n"
        "# Plot 1: Accuracy comparison\n"
        "ax = axes[0, 0]\n"
        "if tff_acc[0] is not None:\n"
        "    ax.plot(rounds, tff_acc, 'b--o', label='TFF FedAvg', markersize=3, alpha=0.7)\n"
        "ax.plot(rounds, enh_acc, 'r-s', label='Enhanced (Ours)', markersize=3, alpha=0.7)\n"
        "ax.set_xlabel('Federated Round')\n"
        "ax.set_ylabel('Accuracy')\n"
        "ax.set_title('TFF FedAvg vs Enhanced Accuracy')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "# Plot 2: Accuracy improvement (delta)\n"
        "ax = axes[0, 1]\n"
        "if tff_acc[0] is not None:\n"
        "    deltas = [e - t for e, t in zip(enh_acc, tff_acc) if t is not None]\n"
        "    ax.bar(rounds[:len(deltas)], deltas, color=['green' if d >= 0 else 'red' for d in deltas], alpha=0.7)\n"
        "    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)\n"
        "    ax.set_xlabel('Federated Round')\n"
        "    ax.set_ylabel('Accuracy Improvement (Δ)')\n"
        "    ax.set_title('Enhanced − FedAvg Accuracy Delta')\n"
        "    ax.grid(True, alpha=0.3)\n"
        "else:\n"
        "    ax.text(0.5, 0.5, 'Comparison mode disabled', ha='center', va='center', transform=ax.transAxes)\n"
        "\n"
        "# Plot 3: Accepted vs Rejected updates\n"
        "ax = axes[1, 0]\n"
        "ax.bar(rounds, history['num_accepted'], label='Accepted', color='green', alpha=0.7)\n"
        "ax.bar(rounds, history['num_rejected'], bottom=history['num_accepted'],\n"
        "       label='Rejected', color='red', alpha=0.7)\n"
        "ax.set_xlabel('Federated Round')\n"
        "ax.set_ylabel('Number of Updates')\n"
        "ax.set_title('Accepted vs Rejected Client Updates')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "# Plot 4: Distillation loss\n"
        "ax = axes[1, 1]\n"
        "kd_losses = history['distillation_loss']\n"
        "kd_rounds = [r for r, l in zip(rounds, kd_losses) if l is not None]\n"
        "kd_values = [l for l in kd_losses if l is not None]\n"
        "if kd_values:\n"
        "    ax.plot(kd_rounds, kd_values, 'g-^', label='KD Loss', markersize=4)\n"
        "    ax.set_xlabel('Federated Round')\n"
        "    ax.set_ylabel('Distillation Loss')\n"
        "    ax.set_title('Knowledge Distillation Loss Over Rounds')\n"
        "    ax.legend()\n"
        "    ax.grid(True, alpha=0.3)\n"
        "else:\n"
        "    ax.text(0.5, 0.5, 'No distillation data', ha='center', va='center', transform=ax.transAxes)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('results_overview.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print('Results plot saved to results_overview.png')\n"
    ))

    cells.append(code_cell(
        "# ── 9b. Reputation Distribution ──────────────────────────────\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "reps = cycle.reputation_ledger.all_reputations()\n"
        "rep_values = list(reps.values())\n"
        "stats = cycle.reputation_ledger.statistics()\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "\n"
        "# Histogram\n"
        "ax = axes[0]\n"
        "ax.hist(rep_values, bins=20, color='steelblue', edgecolor='black', alpha=0.7)\n"
        "ax.axvline(stats['mean_reputation'], color='red', linestyle='--',\n"
        "           label=f\"Mean: {stats['mean_reputation']:.3f}\")\n"
        "ax.axvline(stats['median_reputation'], color='orange', linestyle='--',\n"
        "           label=f\"Median: {stats['median_reputation']:.3f}\")\n"
        "ax.set_xlabel('Reputation Score')\n"
        "ax.set_ylabel('Number of Clients')\n"
        "ax.set_title('Final Client Reputation Distribution')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "# Top/Bottom clients\n"
        "ax = axes[1]\n"
        "ranked = cycle.reputation_ledger.ranked()\n"
        "top_10 = ranked[:10]\n"
        "bottom_10 = ranked[-10:]\n"
        "combined = top_10 + bottom_10\n"
        "names = [c[0] for c in combined]\n"
        "values = [c[1] for c in combined]\n"
        "colors = ['green'] * len(top_10) + ['red'] * len(bottom_10)\n"
        "ax.barh(range(len(combined)), values, color=colors, alpha=0.7)\n"
        "ax.set_yticks(range(len(combined)))\n"
        "ax.set_yticklabels(names, fontsize=7)\n"
        "ax.set_xlabel('Reputation Score')\n"
        "ax.set_title('Top 10 & Bottom 10 Clients by Reputation')\n"
        "ax.grid(True, alpha=0.3)\n"
        "\n"
        "plt.tight_layout()\n"
        "plt.savefig('reputation_distribution.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"
        "print(f'\\nReputation stats:')\n"
        "for k, v in stats.items():\n"
        "    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')\n"
    ))

    cells.append(code_cell(
        "# ── 9c. Round-by-round Summary Table ─────────────────────────\n"
        "print(f\"{'Round':>6} | {'TFF Acc':>8} | {'Enh Acc':>8} | \"\n"
        "      f\"{'Delta':>8} | {'Accepted':>8} | {'Rejected':>8} | {'KD Loss':>10}\")\n"
        "print('-' * 72)\n"
        "\n"
        "for i, rnd in enumerate(history['round']):\n"
        "    tff_a = history['tff_fedavg_accuracy'][i]\n"
        "    enh_a = history['enhanced_accuracy'][i]\n"
        "    tff_str = f'{tff_a:.4f}' if tff_a is not None else '  N/A '\n"
        "    delta = f'{enh_a - tff_a:+.4f}' if tff_a is not None else '  N/A '\n"
        "    kd = history['distillation_loss'][i]\n"
        "    kd_str = f'{kd:.5f}' if kd is not None else '    N/A   '\n"
        "    print(f'{rnd:>6} | {tff_str:>8} | {enh_a:>8.4f} | '\n"
        "          f'{delta:>8} | {history[\"num_accepted\"][i]:>8} | '\n"
        "          f'{history[\"num_rejected\"][i]:>8} | {kd_str:>10}')\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 10: Final Evaluation Report
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 10. Final Evaluation Report\n"
        "\n"
        "Run a comprehensive final evaluation on the test set.\n"
    ))

    cells.append(code_cell(
        "from evaluation_metrics import evaluate_and_report\n"
        "\n"
        "# Full evaluation on the test set\n"
        "final_report = evaluate_and_report(\n"
        "    model=cycle.global_model,\n"
        "    test_data=test_ds,\n"
        "    model_name='effnet_global_tff_final',\n"
        "    reports_dir=config.reports_dir,\n"
        "    federated_round=config.global_rounds,\n"
        "    extra_info={\n"
        "        'framework': 'tensorflow_federated',\n"
        "        'num_devices': config.num_devices,\n"
        "        'total_rounds': config.global_rounds,\n"
        "        'local_epochs': config.local_epochs,\n"
        "        'enhancement_modules': [\n"
        "            'enhanced_client_selection',\n"
        "            'update_validation',\n"
        "            'knowledge_distillation',\n"
        "            'client_reputation_ledger',\n"
        "        ],\n"
        "    },\n"
        "    tag='final',\n"
        ")\n"
        "\n"
        "print('\\n' + '=' * 60)\n"
        "print('  FINAL EVALUATION RESULTS')\n"
        "print('=' * 60)\n"
        "cls = final_report.classification\n"
        "print(f'  Accuracy:         {cls.accuracy:.4f}')\n"
        "print(f'  F1 Score (macro): {cls.f1_macro:.4f}')\n"
        "print(f'  F1 Score (wt.):   {cls.f1_weighted:.4f}')\n"
        "print(f'  Precision:        {cls.precision_macro:.4f}')\n"
        "print(f'  Recall:           {cls.recall_macro:.4f}')\n"
        "print(f'  ROC-AUC:          {cls.roc_auc:.4f}')\n"
        "print(f'\\n  Confusion Matrix:')\n"
        "if cls.confusion_matrix:\n"
        "    print(f'    TN={cls.confusion_matrix[0][0]:>5}  FP={cls.confusion_matrix[0][1]:>5}')\n"
        "    print(f'    FN={cls.confusion_matrix[1][0]:>5}  TP={cls.confusion_matrix[1][1]:>5}')\n"
        "\n"
        "lat = final_report.latency\n"
        "print(f'\\n  Inference Latency:')\n"
        "print(f'    Mean: {lat.mean_ms:.2f} ms | P95: {lat.p95_ms:.2f} ms')\n"
        "\n"
        "sz = final_report.model_size\n"
        "print(f'\\n  Model Size:')\n"
        "print(f'    Parameters: {sz.total_params:,}')\n"
        "print(f'    File size:  {sz.file_size_mb:.2f} MB')\n"
        "print('=' * 60)\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 11: Export & Download
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "## 11. Export & Download Results\n"
        "\n"
        "Download the trained model, TF Lite exports, reports, and\n"
        "reputation ledger.\n"
    ))

    cells.append(code_cell(
        "import os\n"
        "import zipfile\n"
        "\n"
        "# ── Save final Keras model ───────────────────────────────────\n"
        "cycle.global_model.save('effnet_global_tff_trained.h5')\n"
        "print('Saved: effnet_global_tff_trained.h5')\n"
        "\n"
        "# ── List all output files ────────────────────────────────────\n"
        "output_files = []\n"
        "\n"
        "# Model files\n"
        "for f in ['effnet_global_tff_trained.h5',\n"
        "          config.tflite_output_path,\n"
        "          config.tflite_output_path.replace('.tflite', '_quantised.tflite')]:\n"
        "    if os.path.exists(f):\n"
        "        output_files.append(f)\n"
        "\n"
        "# Reports\n"
        "if os.path.isdir('reports'):\n"
        "    for f in os.listdir('reports'):\n"
        "        output_files.append(os.path.join('reports', f))\n"
        "\n"
        "# Visualisation plots\n"
        "for f in ['results_overview.png', 'reputation_distribution.png']:\n"
        "    if os.path.exists(f):\n"
        "        output_files.append(f)\n"
        "\n"
        "print(f'\\nOutput files ({len(output_files)} total):')\n"
        "for f in output_files:\n"
        "    size = os.path.getsize(f) if os.path.exists(f) else 0\n"
        "    print(f'  {f} ({size / 1024:.1f} KB)')\n"
        "\n"
        "# ── Create ZIP archive ───────────────────────────────────────\n"
        "zip_name = 'fl_cycle_results.zip'\n"
        "with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:\n"
        "    for f in output_files:\n"
        "        zf.write(f)\n"
        "print(f'\\nCreated: {zip_name} ({os.path.getsize(zip_name) / 1024:.1f} KB)')\n"
    ))

    cells.append(code_cell(
        "# ── Download results ─────────────────────────────────────────\n"
        "from google.colab import files\n"
        "files.download('fl_cycle_results.zip')\n"
        "print('Download initiated.')\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Section 12: Quick Demo (Reduced Settings)
    # ════════════════════════════════════════════════════════════════════
    cells.append(md_cell(
        "---\n"
        "\n"
        "## Appendix A: Quick Demo (Reduced Settings)\n"
        "\n"
        "If the full 100-device / 50-round cycle takes too long, use this\n"
        "cell for a quick smoke-test with reduced parameters.\n"
    ))

    cells.append(code_cell(
        "# ── Quick Demo: 8 clients, 3 rounds ─────────────────────────\n"
        "# Uncomment and run this cell for a faster test.\n"
        "# Comment out Section 8 above to avoid running the full cycle.\n"
        "\n"
        "# from tff_federated_cycle import TFFFederatedLearningCycle, TFFCycleConfig\n"
        "# from tff_data_utils import generate_synthetic_data, generate_proxy_data, partition_data_iid_tff\n"
        "# from knowledge_distillation import DistillationConfig\n"
        "# import numpy as np, tensorflow as tf\n"
        "#\n"
        "# np.random.seed(42); tf.random.set_seed(42)\n"
        "#\n"
        "# quick_config = TFFCycleConfig(\n"
        "#     model_path='effnet_ffpp_small_data.h5',\n"
        "#     num_devices=8, local_epochs=1, global_rounds=3,\n"
        "#     clients_per_round=4, local_batch_size=16,\n"
        "#     local_lr=1e-3, eval_every=1,\n"
        "#     enable_distillation=True, enable_comparison=True,\n"
        "#     distillation_config=DistillationConfig(\n"
        "#         temperature=3.0, lam=0.7, epochs=2,\n"
        "#         batch_size=16, learning_rate=1e-3),\n"
        "# )\n"
        "#\n"
        "# quick_cycle = TFFFederatedLearningCycle(quick_config)\n"
        "# model = quick_cycle.load_global_model()\n"
        "# input_shape = model.input_shape[1:]\n"
        "# quick_config.input_shape = input_shape\n"
        "#\n"
        "# train = generate_synthetic_data(8 * 30, input_shape, seed=10)\n"
        "# val   = generate_synthetic_data(100, input_shape, seed=20)\n"
        "# test  = generate_synthetic_data(120, input_shape, seed=30)\n"
        "# proxy = generate_proxy_data(80, input_shape, seed=40)\n"
        "# sup   = generate_synthetic_data(60, input_shape, seed=50)\n"
        "#\n"
        "# client_data = partition_data_iid_tff(train, 8)\n"
        "# quick_cycle.create_clients(client_data)\n"
        "# quick_cycle.setup_tff_process()\n"
        "# quick_cycle.setup_enhancement_modules()\n"
        "#\n"
        "# history = quick_cycle.run(\n"
        "#     server_val_data=val, test_data=test,\n"
        "#     proxy_data=proxy, supervised_data=sup,\n"
        "# )\n"
        "#\n"
        "# for r, t_a, e_a in zip(history['round'], history['tff_fedavg_accuracy'], history['enhanced_accuracy']):\n"
        "#     t_s = f'{t_a:.4f}' if t_a else 'N/A'\n"
        "#     print(f'  Round {r}: FedAvg={t_s} | Enhanced={e_a:.4f}')\n"
    ))

    # ════════════════════════════════════════════════════════════════════
    #  Assemble notebook
    # ════════════════════════════════════════════════════════════════════
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "name": "Enhanced_FL_Cycle_DeepFake_Detection.ipynb",
                "gpuType": "T4",
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    return notebook


# ── Main ───────────────────────────────────────────────────────────────

def main():
    output_path = Path(__file__).parent / "Enhanced_FL_Cycle_DeepFake_Detection.ipynb"
    notebook = build_notebook()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    size_kb = output_path.stat().st_size / 1024
    print(f"Notebook generated: {output_path.name} ({size_kb:.0f} KB)")
    num_cells = len(notebook["cells"])
    code_cells = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    md_cells = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    print(f"  Total cells: {num_cells} ({code_cells} code, {md_cells} markdown)")


if __name__ == "__main__":
    main()
