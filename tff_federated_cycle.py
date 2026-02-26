"""
TFF Federated Learning Cycle — Main Orchestrator
==================================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

Integrates **TensorFlow Federated** with all five enhancement modules
into one end-to-end pipeline:

 1. **Enhanced Client Selection**   (``enhanced_client_selection.py``)
 2. **Update Validation & Weighing** (``update_validation.py``)
 3. **Knowledge Distillation**      (``knowledge_distillation.py``)
 4. **Client Reputation Ledger**    (``client_reputation_ledger.py``)
 5. **Evaluation Metrics**          (``evaluation_metrics.py``)

Architecture
------------
TFF handles the **core federated computation**: model broadcasting,
client-side local training, and data-weighted Federated Averaging.
Our thesis enhancements operate as a **post-aggregation refinement
layer** that runs in the outer Python loop after each TFF round.

Per-round pipeline
~~~~~~~~~~~~~~~~~~
1. **Client selection** (Part 1) — multi-criteria scoring to choose
   which clients participate.  Selected client IDs determine which
   data is passed to TFF's ``process.next()``.
2. **TFF round** — ``process.next(state, federated_data)`` performs
   broadcasting, local training (``local_epochs`` via dataset repeat),
   and weighted FedAvg aggregation.
3. **Per-client analysis** — local training is re-run *outside TFF* on
   the selected clients so that per-client model weights are available
   for contribution scoring, distillation, and validation.
4. **Update validation & contribution-weighted aggregation** (Part 2) —
   re-aggregate using contribution weights ``c_i`` (which account for
   validation gain, similarity, data volume, and reputation).  This
   replaces TFF's data-volume-only FedAvg result.
5. **Knowledge distillation** (Part 3) — refine the server model by
   distilling knowledge from the contribution-weighted client ensemble.
6. **Reputation ledger update** (Part 4) — feed validation gains and
   contribution scores into the persistent ledger.
7. **Evaluation & reporting** (Part 5) — periodic full evaluation with
   JSON + text reports.
8. **Inject enhanced weights** back into the TFF server state for the
   next round.

Comparison mode
~~~~~~~~~~~~~~~
When ``enable_comparison=True`` the cycle logs **both** TFF's standard
FedAvg accuracy and our enhanced accuracy each round, providing a
direct side-by-side comparison for the thesis.

Configuration
~~~~~~~~~~~~~
- **Model**: ``effnet_ffpp_small_data.h5`` (EfficientNet, binary)
- **Devices**: 100 simulated clients
- **Local epochs**: 5 per round
- **Global rounds**: 50
- **Frameworks**: TensorFlow Federated + TF Lite

Environment
-----------
Requires ``tensorflow-federated >= 0.48.0``.
See ``requirements_tff.txt`` for the exact compatible stack.
Recommended runtime: **Google Colab**.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ---------- Conditional TFF import ------------------------------------- #
try:
    import tensorflow_federated as tff
    TFF_AVAILABLE = True
except ImportError:
    tff = None  # type: ignore[assignment]
    TFF_AVAILABLE = False

# ---------- Existing modules  (Parts 1–5) ----------------------------- #
from enhanced_client_selection import (          # Part 1
    ClientMetrics,
    FederatedClient,
    ReputationLedger,
    SelectionWeights,
    EnhancedClientSelector,
)
from update_validation import (                  # Part 2
    ContributionWeights,
    ClippingConfig,
    ClientUpdateRecord,
    UpdateValidator,
)
from knowledge_distillation import (             # Part 3
    DistillationConfig,
    run_distillation_round,
)
from client_reputation_ledger import (           # Part 4
    ReputationConfig,
    ClientReputationLedger,
    update_ledger_from_records,
)
from evaluation_metrics import (                 # Part 5
    FederatedModelEvaluator,
    evaluate_and_report,
)

# ---------- TFF wrappers ---------------------------------------------- #
from tff_data_utils import (
    TFFDataManager,
    _require_tff,
    partition_data_iid_tff,
    generate_synthetic_data,
    generate_proxy_data,
)
from tff_learning_process import (
    TFFModelFactory,
    TFFProcessConfig,
    build_tff_learning_process,
    TFFRoundExecutor,
    keras_weights_to_tff,
    tff_weights_to_keras,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ====================================================================== #
#  1.  CONFIGURATION                                                      #
# ====================================================================== #

@dataclass
class TFFCycleConfig:
    """
    Central configuration for the TFF-based Federated Learning cycle.
    """
    # -- Core FL settings ---------------------------------------------- #
    model_path: str = "effnet_ffpp_small_data.h5"
    num_devices: int = 100
    local_epochs: int = 5
    global_rounds: int = 50
    clients_per_round: int = 15
    local_batch_size: int = 32
    local_lr: float = 1e-4
    server_lr: float = 1.0
    eval_every: int = 5

    # -- TFF process settings ------------------------------------------ #
    client_optimizer: str = "adam"
    server_optimizer: str = "sgd"

    # -- Comparison mode ----------------------------------------------- #
    enable_comparison: bool = True     # Log TFF FedAvg vs enhanced

    # -- Distillation (Part 3) ---------------------------------------- #
    enable_distillation: bool = True
    distillation_config: DistillationConfig = field(
        default_factory=lambda: DistillationConfig(
            temperature=3.0,
            lam=0.7,
            epochs=3,
            batch_size=32,
            learning_rate=1e-4,
        )
    )

    # -- Client selection (Part 1) ------------------------------------- #
    selection_weights: SelectionWeights = field(
        default_factory=lambda: SelectionWeights(
            w_v=0.30,
            w_d=0.20,
            w_l=0.10,
            w_r=0.25,
            w_s=0.15,
        )
    )

    # -- Update validation (Part 2) ----------------------------------- #
    contribution_weights: ContributionWeights = field(
        default_factory=lambda: ContributionWeights(
            alpha=0.35,
            beta=0.20,
            gamma=0.20,
            delta=0.25,
        )
    )
    clipping_config: ClippingConfig = field(
        default_factory=lambda: ClippingConfig(
            clip_threshold=10.0,
            clip_value=5.0,
        )
    )
    harmful_threshold: float = 0.02

    # -- Reputation (Part 4) ------------------------------------------ #
    reputation_config: ReputationConfig = field(
        default_factory=lambda: ReputationConfig(
            theta=0.0,
            gamma=0.10,
            decay_rate=0.99,
            floor=0.05,
            ceiling=1.0,
            initial_reputation=0.50,
            penalty_factor=0.05,
        )
    )

    # -- Evaluation & output (Part 5) --------------------------------- #
    reports_dir: str = "reports"
    tflite_output_path: str = "effnet_global_tff_final.tflite"
    input_shape: Tuple[int, ...] = (224, 224, 3)


# ====================================================================== #
#  2.  TF LITE CONVERSION                                                 #
# ====================================================================== #

def convert_to_tflite(
    model: tf.keras.Model,
    output_path: str,
    quantise: bool = False,
) -> str:
    """Convert a Keras model to TF Lite format."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantise:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    Path(output_path).write_bytes(tflite_bytes)
    size_mb = len(tflite_bytes) / (1024 * 1024)
    logger.info(
        "TF Lite model saved → %s  (%.2f MB, quantised=%s)",
        output_path, size_mb, quantise,
    )
    return output_path


# ====================================================================== #
#  3.  TFF FEDERATED LEARNING CYCLE                                       #
# ====================================================================== #

class TFFFederatedLearningCycle:
    """
    End-to-end Federated Learning cycle using TFF,
    integrating all five enhancement modules.

    See module docstring for the detailed per-round pipeline.
    """

    def __init__(self, config: Optional[TFFCycleConfig] = None) -> None:
        self.config = config or TFFCycleConfig()
        self.global_model: Optional[tf.keras.Model] = None
        self.clients: List[FederatedClient] = []
        self.client_datasets: Dict[str, tf.data.Dataset] = {}

        # TFF components
        self.data_manager: Optional[TFFDataManager] = None
        self.tff_executor: Optional[TFFRoundExecutor] = None

        # Enhancement components (Parts 1–5)
        self.reputation_ledger: Optional[ClientReputationLedger] = None
        self.basic_ledger: Optional[ReputationLedger] = None
        self.selector: Optional[EnhancedClientSelector] = None
        self.validator: Optional[UpdateValidator] = None
        self.evaluator: Optional[FederatedModelEvaluator] = None

        # History
        self.history: Dict[str, list] = {
            "round": [],
            "tff_fedavg_accuracy": [],
            "enhanced_accuracy": [],
            "selected_clients": [],
            "num_accepted": [],
            "num_rejected": [],
            "distillation_loss": [],
        }

    # ------------------------------------------------------------------ #
    #  Initialisation                                                     #
    # ------------------------------------------------------------------ #

    def load_global_model(self) -> tf.keras.Model:
        """Load the pre-trained EfficientNet model."""
        cfg = self.config
        logger.info("Loading global model from %s …", cfg.model_path)
        model = tf.keras.models.load_model(cfg.model_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.local_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(
            "Global model loaded — %s params, input shape %s",
            f"{model.count_params():,}", model.input_shape,
        )
        self.global_model = model
        return model

    def create_clients(
        self,
        client_data: Dict[str, tf.data.Dataset],
    ) -> List[FederatedClient]:
        """Create ``FederatedClient`` objects and store the dataset dict."""
        rng = np.random.RandomState(42)
        clients: List[FederatedClient] = []

        for cid, local_ds in client_data.items():
            n_samples = sum(1 for _ in local_ds)
            metrics = ClientMetrics(
                local_validation_metric=float(rng.uniform(0.4, 0.9)),
                data_volume=n_samples,
                inference_latency=float(rng.uniform(0.01, 0.15)),
                last_selected_round=0,
            )
            clients.append(FederatedClient(
                client_id=cid,
                local_data=local_ds,
                metrics=metrics,
            ))

        logger.info("Created %d federated clients.", len(clients))
        self.clients = clients
        self.client_datasets = client_data
        return clients

    def setup_tff_process(self) -> None:
        """
        Build the TFF learning process and initialise the round executor.
        """
        _require_tff()
        cfg = self.config
        assert self.global_model is not None, "Call load_global_model() first."

        # Data manager
        input_shape = self.global_model.input_shape[1:]
        cfg.input_shape = input_shape
        self.data_manager = TFFDataManager(input_shape=input_shape)
        element_spec = self.data_manager.get_element_spec()

        # Model factory → model_fn
        factory = TFFModelFactory(
            keras_model=self.global_model,
            input_spec=element_spec,
        )
        model_fn = factory.create_model_fn()

        # Build TFF learning process
        process = build_tff_learning_process(
            model_fn=model_fn,
            config=TFFProcessConfig(
                client_lr=cfg.local_lr,
                server_lr=cfg.server_lr,
                client_optimizer=cfg.client_optimizer,
                server_optimizer=cfg.server_optimizer,
            ),
        )

        # Round executor
        self.tff_executor = TFFRoundExecutor(process, self.global_model)
        self.tff_executor.initialize()
        self.tff_executor.inject_pretrained_weights()

        logger.info("TFF process initialised with pre-trained weights.")

    def setup_enhancement_modules(self) -> None:
        """Wire Parts 1–5 (same logic as federated_learning_cycle.py)."""
        cfg = self.config
        assert self.global_model is not None
        assert len(self.clients) > 0

        # Part 4: Reputation ledger
        self.reputation_ledger = ClientReputationLedger(
            config=cfg.reputation_config,
        )
        for c in self.clients:
            self.reputation_ledger.register(c.client_id)
        self.basic_ledger = self.reputation_ledger.as_basic_ledger()

        # Part 1: Client selector
        self.selector = EnhancedClientSelector(
            clients=self.clients,
            reputation_ledger=self.basic_ledger,
            weights=cfg.selection_weights,
            target_k=cfg.clients_per_round,
        )

        # Part 2: Update validator
        self.validator = UpdateValidator(
            global_model=self.global_model,
            reputation_ledger=self.basic_ledger,
            weights=cfg.contribution_weights,
            clipping=cfg.clipping_config,
            harmful_threshold=cfg.harmful_threshold,
            batch_size=cfg.local_batch_size,
        )

        # Part 5: Evaluator
        self.evaluator = FederatedModelEvaluator(
            model=self.global_model,
            model_name="effnet_global_tff",
            reports_dir=cfg.reports_dir,
        )

        logger.info("Enhancement modules (Parts 1–5) initialised.")

    # ------------------------------------------------------------------ #
    #  Local training (manual — for per-client analysis)                  #
    # ------------------------------------------------------------------ #

    def _local_train(
        self,
        client: FederatedClient,
        global_weights: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], int]:
        """
        Manual local training (outside TFF) to obtain per-client model
        weights for contribution scoring and distillation.
        """
        cfg = self.config
        local_model = tf.keras.models.clone_model(self.global_model)
        local_model.build(self.global_model.input_shape)
        local_model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.local_lr),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        local_model.set_weights(global_weights)

        if client.local_data is None:
            return global_weights, 0

        dataset = client.local_data.batch(cfg.local_batch_size)
        local_model.fit(dataset, epochs=cfg.local_epochs, verbose=0)
        return local_model.get_weights(), client.metrics.data_volume

    # ------------------------------------------------------------------ #
    #  Reputation sync                                                    #
    # ------------------------------------------------------------------ #

    def _sync_reputation_to_basic_ledger(self) -> None:
        """Copy extended ledger → basic ledger used by Parts 1 & 2."""
        updated_basic = self.reputation_ledger.as_basic_ledger()
        self.basic_ledger._scores = updated_basic._scores

    # ------------------------------------------------------------------ #
    #  Single round                                                       #
    # ------------------------------------------------------------------ #

    def execute_round(
        self,
        current_round: int,
        server_val_data: tf.data.Dataset,
        proxy_data: Optional[tf.data.Dataset] = None,
        supervised_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, Any]:
        """
        Execute one complete TFF + enhanced federated round.

        Returns a summary dict with both TFF FedAvg and enhanced accuracy.
        """
        cfg = self.config

        logger.info("=" * 70)
        logger.info("  TFF ROUND %d / %d", current_round, cfg.global_rounds)
        logger.info("=" * 70)

        # ── 1. Client selection  (Part 1) ──────────────────────────── #
        selected: List[FederatedClient] = self.selector.select(
            current_round=current_round,
        )
        selected_ids = [c.client_id for c in selected]
        logger.info(
            "Selected %d / %d clients: %s",
            len(selected), len(self.clients), selected_ids,
        )

        # Save pre-round global weights for analysis
        global_weights_before = self.global_model.get_weights()

        # ── 2. TFF federated round (FedAvg baseline) ──────────────── #
        federated_data = self.data_manager.make_federated_data(
            self.client_datasets,
            selected_ids,
            batch_size=cfg.local_batch_size,
            local_epochs=cfg.local_epochs,
        )
        tff_metrics = self.tff_executor.execute_round(federated_data)

        # Extract TFF-aggregated weights
        tff_aggregated_weights = self.tff_executor.get_keras_weights()

        # Quick TFF FedAvg accuracy check
        tff_acc = None
        if cfg.enable_comparison:
            tff_model_tmp = tf.keras.models.clone_model(self.global_model)
            tff_model_tmp.build(self.global_model.input_shape)
            tff_model_tmp.compile(
                optimizer="adam", loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            tff_model_tmp.set_weights(tff_aggregated_weights)
            tff_result = tff_model_tmp.evaluate(
                server_val_data.batch(cfg.local_batch_size),
                verbose=0, return_dict=True,
            )
            tff_acc = tff_result.get("accuracy", 0.0)
            logger.info(
                "TFF FedAvg accuracy (round %d): %.4f",
                current_round, tff_acc,
            )

        # ── 3. Per-client analysis (manual, for Parts 2/3/4) ──────── #
        client_updates: Dict[str, List[np.ndarray]] = {}
        data_volumes: Dict[str, int] = {}

        for client in selected:
            updated_w, n = self._local_train(client, global_weights_before)
            client_updates[client.client_id] = updated_w
            data_volumes[client.client_id] = max(n, 1)
            client.metrics.last_selected_round = current_round

        # ── 4. Update validation & contribution aggregation (Part 2) ─ #
        records: List[ClientUpdateRecord] = self.validator.validate_updates(
            client_updates=client_updates,
            data_volumes=data_volumes,
            server_val_data=server_val_data,
        )
        enhanced_weights = self.validator.aggregate_weighted(
            records, global_weights_before,
        )

        num_accepted = sum(1 for r in records if not r.rejected)
        num_rejected = sum(1 for r in records if r.rejected)
        logger.info(
            "Updates: %d accepted, %d rejected out of %d.",
            num_accepted, num_rejected, len(records),
        )

        # Apply enhanced weights to global model
        self.global_model.set_weights(enhanced_weights)
        self.validator.global_model.set_weights(enhanced_weights)

        # ── 5. Knowledge distillation  (Part 3) ───────────────────── #
        distill_loss = None
        if cfg.enable_distillation and proxy_data is not None:
            contribution_weights = {
                r.client_id: r.contribution_weight
                for r in records
                if not r.rejected and r.contribution_weight > 0
            }

            if len(contribution_weights) >= 2:
                logger.info(
                    "Running knowledge distillation with %d teacher(s) …",
                    len(contribution_weights),
                )
                kd_history = run_distillation_round(
                    global_model=self.global_model,
                    client_weights={
                        cid: client_updates[cid]
                        for cid in contribution_weights
                    },
                    contribution_weights=contribution_weights,
                    proxy_data=proxy_data,
                    supervised_data=supervised_data,
                    config=cfg.distillation_config,
                )
                distill_loss = kd_history["loss_total"][-1]
                logger.info(
                    "Distillation complete — final loss %.5f", distill_loss,
                )

        # ── 6. Reputation update  (Part 4) ────────────────────────── #
        update_ledger_from_records(
            self.reputation_ledger, records, current_round,
        )
        self.validator.update_reputations(records)
        self._sync_reputation_to_basic_ledger()

        # ── 7. Enhanced accuracy check ────────────────────────────── #
        enhanced_result = self.global_model.evaluate(
            server_val_data.batch(cfg.local_batch_size),
            verbose=0, return_dict=True,
        )
        enhanced_acc = enhanced_result.get("accuracy", 0.0)

        if cfg.enable_comparison and tff_acc is not None:
            delta = enhanced_acc - tff_acc
            logger.info(
                "Round %d — TFF FedAvg: %.4f | Enhanced: %.4f | Δ=%+.4f",
                current_round, tff_acc, enhanced_acc, delta,
            )
        else:
            logger.info(
                "Round %d — Enhanced accuracy: %.4f",
                current_round, enhanced_acc,
            )

        # ── 8. Inject enhanced weights into TFF state ─────────────── #
        self.tff_executor.set_keras_weights(
            self.global_model.get_weights(),
        )

        return {
            "round": current_round,
            "selected": selected_ids,
            "tff_fedavg_accuracy": tff_acc,
            "enhanced_accuracy": enhanced_acc,
            "num_accepted": num_accepted,
            "num_rejected": num_rejected,
            "records": records,
            "distillation_loss": distill_loss,
            "tff_metrics": tff_metrics,
        }

    # ------------------------------------------------------------------ #
    #  Full cycle                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        server_val_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
        proxy_data: Optional[tf.data.Dataset] = None,
        supervised_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, list]:
        """
        Run the full TFF Federated Learning cycle.

        Parameters
        ----------
        server_val_data : tf.data.Dataset
            Server validation set for update scoring.
        test_data : tf.data.Dataset
            Independent test set for full evaluation (Part 5).
        proxy_data : tf.data.Dataset | None
            Unlabelled proxy data for distillation (Part 3).
        supervised_data : tf.data.Dataset | None
            Labelled data for combined distillation loss (Part 3).

        Returns
        -------
        history : dict
        """
        cfg = self.config

        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║  TFF FEDERATED LEARNING CYCLE — START                   ║")
        logger.info("║  Devices: %3d | Rounds: %3d | Local epochs: %d          ║",
                     cfg.num_devices, cfg.global_rounds, cfg.local_epochs)
        logger.info("║  TFF FedAvg: ON | Comparison: %-4s                      ║",
                     "ON" if cfg.enable_comparison else "OFF")
        logger.info("╚══════════════════════════════════════════════════════════╝")

        # -- Baseline evaluation --------------------------------------- #
        logger.info("Evaluating baseline model …")
        baseline_report = self.evaluator.evaluate(
            test_data=test_data,
            batch_size=cfg.local_batch_size,
            federated_round=0,
            extra_info={"stage": "baseline", "framework": "tff"},
        )
        self.evaluator.save_report(baseline_report, tag="round_000_baseline_tff")
        logger.info(
            "Baseline — Acc: %.4f, F1: %.4f, AUC: %.4f",
            baseline_report.classification.accuracy,
            baseline_report.classification.f1_macro,
            baseline_report.classification.roc_auc,
        )
        all_reports = [baseline_report]
        cycle_start = time.time()

        # ============================================================== #
        #  MAIN LOOP                                                      #
        # ============================================================== #

        for t in range(1, cfg.global_rounds + 1):
            round_start = time.time()

            info = self.execute_round(
                current_round=t,
                server_val_data=server_val_data,
                proxy_data=proxy_data,
                supervised_data=supervised_data,
            )

            # Record history
            self.history["round"].append(t)
            self.history["tff_fedavg_accuracy"].append(info["tff_fedavg_accuracy"])
            self.history["enhanced_accuracy"].append(info["enhanced_accuracy"])
            self.history["selected_clients"].append(info["selected"])
            self.history["num_accepted"].append(info["num_accepted"])
            self.history["num_rejected"].append(info["num_rejected"])
            self.history["distillation_loss"].append(info["distillation_loss"])

            round_elapsed = time.time() - round_start
            logger.info("Round %d completed in %.1fs.", t, round_elapsed)

            # -- Periodic full evaluation (Part 5) --------------------- #
            is_eval_round = (
                t % cfg.eval_every == 0
                or t == 1
                or t == cfg.global_rounds
            )
            if is_eval_round:
                logger.info("Full evaluation (round %d) …", t)
                report = self.evaluator.evaluate(
                    test_data=test_data,
                    batch_size=cfg.local_batch_size,
                    federated_round=t,
                    extra_info={
                        "framework": "tff",
                        "tff_fedavg_acc": info["tff_fedavg_accuracy"],
                        "enhanced_acc": info["enhanced_accuracy"],
                        "accepted": info["num_accepted"],
                        "rejected": info["num_rejected"],
                    },
                )
                self.evaluator.save_report(report, tag=f"tff_round_{t:03d}")
                all_reports.append(report)
                logger.info(
                    "Round %d eval — Acc: %.4f, F1: %.4f, AUC: %.4f",
                    t,
                    report.classification.accuracy,
                    report.classification.f1_macro,
                    report.classification.roc_auc,
                )

        total_elapsed = time.time() - cycle_start

        # ============================================================== #
        #  POST-CYCLE                                                     #
        # ============================================================== #

        logger.info("╔══════════════════════════════════════════════════════════╗")
        logger.info("║  TFF FEDERATED LEARNING CYCLE — COMPLETE                ║")
        logger.info("║  Total time: %.1fs                                      ║", total_elapsed)
        logger.info("╚══════════════════════════════════════════════════════════╝")

        # Comparison report
        if len(all_reports) > 1:
            self.evaluator.save_comparison_report(all_reports)

        # Save reputation ledger
        ledger_path = Path(cfg.reports_dir) / "reputation_ledger_tff_final.json"
        self.reputation_ledger.save(str(ledger_path))

        # TF Lite export
        convert_to_tflite(self.global_model, cfg.tflite_output_path, quantise=False)
        convert_to_tflite(
            self.global_model,
            cfg.tflite_output_path.replace(".tflite", "_quantised.tflite"),
            quantise=True,
        )

        # Final summary
        self._print_summary()

        return self.history

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #

    def _print_summary(self) -> None:
        """Pretty-print a final training summary."""
        h = self.history
        if not h["round"]:
            return

        best_idx = int(np.argmax(h["enhanced_accuracy"]))
        best_round = h["round"][best_idx]
        best_acc = h["enhanced_accuracy"][best_idx]
        final_acc = h["enhanced_accuracy"][-1]

        stats = self.reputation_ledger.statistics()

        print("\n" + "=" * 64)
        print("  TFF FEDERATED LEARNING CYCLE — FINAL SUMMARY")
        print("=" * 64)
        print(f"  Total rounds:              {len(h['round'])}")
        print(f"  Best enhanced accuracy:    {best_acc:.4f}  (round {best_round})")
        print(f"  Final enhanced accuracy:   {final_acc:.4f}")
        print(f"  Mean enhanced accuracy:    {np.mean(h['enhanced_accuracy']):.4f}")

        if h["tff_fedavg_accuracy"][0] is not None:
            mean_tff = np.mean([a for a in h["tff_fedavg_accuracy"] if a is not None])
            print(f"  Mean TFF FedAvg accuracy:  {mean_tff:.4f}")
            deltas = [
                e - t for e, t in zip(h["enhanced_accuracy"], h["tff_fedavg_accuracy"])
                if t is not None
            ]
            if deltas:
                print(f"  Mean improvement (Δ):      {np.mean(deltas):+.4f}")

        print(f"  Total accepted updates:    {sum(h['num_accepted'])}")
        print(f"  Total rejected updates:    {sum(h['num_rejected'])}")
        print(f"  Reputation — mean:         {stats.get('mean_reputation', 0):.4f}")
        print(f"  Reputation — std:          {stats.get('std_reputation', 0):.4f}")

        kd = [l for l in h["distillation_loss"] if l is not None]
        if kd:
            print(f"  Avg distillation loss:     {np.mean(kd):.5f}")

        print(f"  TF Lite model:             {self.config.tflite_output_path}")
        print("=" * 64 + "\n")


# ====================================================================== #
#  4.  ENTRY POINT  (full production run with .h5 model)                  #
# ====================================================================== #

def main() -> None:
    """
    Full TFF FL cycle with ``effnet_ffpp_small_data.h5``.

    Requires TFF and a compatible TF version.
    """
    _require_tff()

    np.random.seed(42)
    tf.random.set_seed(42)

    config = TFFCycleConfig(
        model_path="effnet_ffpp_small_data.h5",
        num_devices=100,
        local_epochs=5,
        global_rounds=50,
        clients_per_round=15,
        local_batch_size=32,
        local_lr=1e-4,
        server_lr=1.0,
        eval_every=10,
        enable_distillation=True,
        enable_comparison=True,
    )

    cycle = TFFFederatedLearningCycle(config)

    # Load model
    model = cycle.load_global_model()
    input_shape = model.input_shape[1:]
    config.input_shape = input_shape

    # Synthetic data (replace with real FF++ c23 loaders)
    logger.info("Generating synthetic data …")
    TOTAL = config.num_devices * 10
    train_ds = generate_synthetic_data(TOTAL, input_shape, seed=1)
    val_ds = generate_synthetic_data(200, input_shape, seed=2)
    test_ds = generate_synthetic_data(300, input_shape, seed=3)
    proxy_ds = generate_proxy_data(150, input_shape, seed=4)
    sup_ds = generate_synthetic_data(100, input_shape, seed=5)

    client_data = partition_data_iid_tff(train_ds, config.num_devices)

    # Build cycle
    cycle.create_clients(client_data)
    cycle.setup_tff_process()
    cycle.setup_enhancement_modules()

    # Run
    history = cycle.run(
        server_val_data=val_ds,
        test_data=test_ds,
        proxy_data=proxy_ds,
        supervised_data=sup_ds,
    )

    logger.info("Done. History keys: %s", list(history.keys()))


# ====================================================================== #
#  DEMO / SMOKE-TEST                                                      #
# ====================================================================== #

if __name__ == "__main__":
    print("\n===  TFF Federated Learning Cycle — Demo  ===\n")

    np.random.seed(42)
    tf.random.set_seed(42)

    # ---- 1. Build a tiny model --------------------------------------- #
    INPUT_DIM = 16
    demo_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    demo_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Demo model — {demo_model.count_params()} params")

    # ---- 2. Config (reduced) ---------------------------------------- #
    demo_config = TFFCycleConfig(
        model_path="(in-memory demo)",
        num_devices=8,
        local_epochs=1,
        global_rounds=3,
        clients_per_round=4,
        local_batch_size=16,
        local_lr=1e-3,
        server_lr=1.0,
        eval_every=1,
        enable_distillation=True,
        enable_comparison=True,
        distillation_config=DistillationConfig(
            temperature=3.0, lam=0.7, epochs=2,
            batch_size=16, learning_rate=1e-3,
        ),
    )

    # ---- 3. Synthetic data ------------------------------------------- #
    input_shape = (INPUT_DIM,)
    N_CLI = demo_config.num_devices
    TOTAL = N_CLI * 30

    train_ds = generate_synthetic_data(TOTAL, input_shape, seed=10)
    val_ds = generate_synthetic_data(100, input_shape, seed=20)
    test_ds = generate_synthetic_data(120, input_shape, seed=30)
    proxy_ds = generate_proxy_data(80, input_shape, seed=40)
    sup_ds = generate_synthetic_data(60, input_shape, seed=50)
    client_data = partition_data_iid_tff(train_ds, N_CLI)

    # ---- 4. Build cycle ---------------------------------------------- #
    cycle = TFFFederatedLearningCycle(demo_config)
    cycle.global_model = demo_model

    cycle.create_clients(client_data)

    # ---- 5. TFF-dependent vs fallback -------------------------------- #
    if TFF_AVAILABLE:
        print("\n--- TFF available — running full TFF demo ---\n")
        cycle.setup_tff_process()
        cycle.setup_enhancement_modules()

        history = cycle.run(
            server_val_data=val_ds,
            test_data=test_ds,
            proxy_data=proxy_ds,
            supervised_data=sup_ds,
        )

        print("\nRound-by-round comparison:")
        for rnd, tff_a, enh_a in zip(
            history["round"],
            history["tff_fedavg_accuracy"],
            history["enhanced_accuracy"],
        ):
            tff_str = f"{tff_a:.4f}" if tff_a is not None else "  N/A "
            print(f"  Round {rnd}: TFF FedAvg={tff_str} | Enhanced={enh_a:.4f}")

        # Cleanup demo TFLite files
        for p in [demo_config.tflite_output_path,
                  demo_config.tflite_output_path.replace(".tflite", "_quantised.tflite")]:
            Path(p).unlink(missing_ok=True)

    else:
        print(
            "⚠  TFF not installed — running enhancement-only demo.\n"
            "   This exercises Parts 1–5 without TFF's federated layer.\n"
            "   For full TFF integration, see requirements_tff.txt.\n"
        )

        # Fallback: run the non-TFF orchestrator logic as a sanity check
        cycle.setup_enhancement_modules()

        # Simulate 3 rounds using manual local training + enhancements
        for t in range(1, demo_config.global_rounds + 1):
            selected = cycle.selector.select(current_round=t)
            selected_ids = [c.client_id for c in selected]
            global_w = cycle.global_model.get_weights()

            client_updates = {}
            data_volumes = {}
            for c in selected:
                w, n = cycle._local_train(c, global_w)
                client_updates[c.client_id] = w
                data_volumes[c.client_id] = max(n, 1)
                c.metrics.last_selected_round = t

            records = cycle.validator.validate_updates(
                client_updates, data_volumes, val_ds,
            )
            new_w = cycle.validator.aggregate_weighted(records, global_w)
            cycle.global_model.set_weights(new_w)
            cycle.validator.global_model.set_weights(new_w)

            # Distillation
            cw = {r.client_id: r.contribution_weight
                  for r in records if not r.rejected and r.contribution_weight > 0}
            if len(cw) >= 2:
                run_distillation_round(
                    cycle.global_model,
                    {cid: client_updates[cid] for cid in cw},
                    cw, proxy_ds, sup_ds,
                    demo_config.distillation_config,
                )

            update_ledger_from_records(cycle.reputation_ledger, records, t)
            cycle.validator.update_reputations(records)
            cycle._sync_reputation_to_basic_ledger()

            acc = cycle.global_model.evaluate(
                val_ds.batch(16), verbose=0, return_dict=True,
            ).get("accuracy", 0.0)
            print(f"  Round {t}: Enhanced accuracy = {acc:.4f} "
                  f"(selected: {selected_ids})")

        # Quick evaluation
        report = evaluate_and_report(
            cycle.global_model, test_ds,
            model_name="demo_no_tff", federated_round=t,
        )
        print(f"\nFinal — Acc: {report.classification.accuracy:.4f}, "
              f"F1: {report.classification.f1_macro:.4f}, "
              f"AUC: {report.classification.roc_auc:.4f}")

    print("\nDone.")
