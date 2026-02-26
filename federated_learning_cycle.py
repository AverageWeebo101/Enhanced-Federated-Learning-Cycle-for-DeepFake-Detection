"""
Federated Learning Cycle — Main Orchestrator
==============================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

Integrates **all five** modules into one end-to-end pipeline:

 1. **Enhanced Client Selection**  (``enhanced_client_selection.py``)
 2. **Update Validation & Contribution Weighing** (``update_validation.py``)
 3. **Server-Side Knowledge Distillation**  (``knowledge_distillation.py``)
 4. **Client Reputation & Ledger**  (``client_reputation_ledger.py``)
 5. **Evaluation Metrics & Reporting**  (``evaluation_metrics.py``)

Cycle summary (per round)
-------------------------
1.  Select clients from the pool via multi-criteria scoring  (Part 1).
2.  Distribute global weights → clients train locally for 5 epochs.
3.  Validate updates, assign contribution weights, reject harmful ones,
    and perform weighted aggregation  (Part 2).
4.  Optionally refine the aggregated model with server-side knowledge
    distillation from the contribution-weighted client ensemble  (Part 3).
5.  Update the persistent reputation ledger based on gains & c_i  (Part 4).
6.  Every ``eval_every`` rounds, run full evaluation & save reports  (Part 5).
7.  After all rounds, convert the final model to TensorFlow Lite.

Configuration
-------------
- **Model**: ``effnet_ffpp_small_data.h5``   (EfficientNet‑based binary
  classifier — Real vs Fake)
- **Devices**: 100 simulated federated clients
- **Local epochs per round**: 5
- **Global aggregation rounds**: 50
- **Frameworks**: TensorFlow / TF Lite / TF Federated concepts
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ====================================================================== #
#  Module imports  (Parts 1–5)                                            #
# ====================================================================== #

from enhanced_client_selection import (       # Part 1
    ClientMetrics,
    FederatedClient,
    ReputationLedger,
    SelectionWeights,
    EnhancedClientSelector,
)
from update_validation import (               # Part 2
    ContributionWeights,
    ClippingConfig,
    ClientUpdateRecord,
    UpdateValidator,
    flatten_weights,
    unflatten_weights,
    compute_update_delta,
    apply_update,
)
from knowledge_distillation import (          # Part 3
    DistillationConfig,
    run_distillation_round,
)
from client_reputation_ledger import (        # Part 4
    ReputationConfig,
    ClientReputationLedger,
    update_ledger_from_records,
)
from evaluation_metrics import (              # Part 5
    FederatedModelEvaluator,
    evaluate_and_report,
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
class FLCycleConfig:
    """
    Central configuration for the entire Federated Learning cycle.

    Parameters
    ----------
    model_path : str
        Path to the pre-trained EfficientNet HDF5 model.
    num_devices : int
        Total number of simulated federated client devices.
    local_epochs : int
        Number of local training epochs per client per round.
    global_rounds : int
        Total number of federated aggregation rounds.
    clients_per_round : int
        Target number of clients selected each round (``k``).
    local_batch_size : int
        Batch size for client-side local training.
    local_lr : float
        Client-side optimiser learning rate.
    eval_every : int
        Run full evaluation every N rounds (also round 1 & last).
    enable_distillation : bool
        Whether to run server-side knowledge distillation each round.
    distillation_config : DistillationConfig
        Hyper-parameters for the distillation loop.
    selection_weights : SelectionWeights
        Weights for the multi-criteria client scoring  (Part 1).
    contribution_weights : ContributionWeights
        α, β, γ, δ for the contribution scoring  (Part 2).
    clipping_config : ClippingConfig
        Norm-clipping parameters for update validation.
    harmful_threshold : float
        ε — reject updates with ``G_i < −ε``.
    reputation_config : ReputationConfig
        Parameters for the persistent reputation ledger  (Part 4).
    reports_dir : str
        Directory for saving evaluation reports  (Part 5).
    tflite_output_path : str
        Path for the final TF Lite model export.
    input_shape : tuple
        Input shape expected by the model  (H, W, C).
    """
    # -- Core FL settings ---------------------------------------------- #
    model_path: str = "effnet_ffpp_small_data.h5"
    num_devices: int = 100
    local_epochs: int = 5
    global_rounds: int = 50
    clients_per_round: int = 15
    local_batch_size: int = 32
    local_lr: float = 1e-4
    eval_every: int = 5

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
            alpha=0.35,   # validation gain
            beta=0.20,    # similarity
            gamma=0.20,   # data volume
            delta=0.25,   # reputation
        )
    )
    clipping_config: ClippingConfig = field(
        default_factory=lambda: ClippingConfig(
            clip_threshold=10.0,
            clip_value=5.0,
        )
    )
    harmful_threshold: float = 0.02

    # -- Reputation ledger (Part 4) ------------------------------------ #
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
    tflite_output_path: str = "effnet_global_final.tflite"
    input_shape: Tuple[int, ...] = (224, 224, 3)


# ====================================================================== #
#  2.  DATA HELPERS  (simulation — replace with real FF++ loaders)        #
# ====================================================================== #

def _generate_synthetic_data(
    num_samples: int,
    input_shape: Tuple[int, ...],
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Generate a synthetic (random) labelled dataset for smoke-testing.

    In production, replace this with a real FF++ c23 data loader that
    returns ``(image, label)`` pairs as ``tf.data.Dataset``.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, *input_shape).astype(np.float32) * 0.1
    y = rng.randint(0, 2, size=(num_samples,)).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((x, y))


def _generate_proxy_data(
    num_samples: int,
    input_shape: Tuple[int, ...],
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """
    Generate unlabelled proxy data (images only) for knowledge
    distillation.  Replace with real FF++ c23 unlabelled images.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, *input_shape).astype(np.float32) * 0.1
    return tf.data.Dataset.from_tensor_slices(x)


def partition_data_iid(
    full_dataset: tf.data.Dataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[str, tf.data.Dataset]:
    """
    IID partition: shuffle the dataset and split evenly across clients.

    Returns
    -------
    dict[str, tf.data.Dataset]
        ``{client_id: local_dataset}``
    """
    all_data = list(full_dataset.shuffle(buffer_size=10_000, seed=seed))
    total = len(all_data)
    shard_size = max(1, total // num_clients)

    partitions: Dict[str, tf.data.Dataset] = {}
    for i in range(num_clients):
        cid = f"client_{i:03d}"
        start = i * shard_size
        end = min(start + shard_size, total)
        if start >= total:
            # Wrap around for the last few clients
            start = start % total
            end = start + shard_size

        shard_x = [elem[0].numpy() for elem in all_data[start:end]]
        shard_y = [elem[1].numpy() for elem in all_data[start:end]]

        if len(shard_x) == 0:
            shard_x = [all_data[0][0].numpy()]
            shard_y = [all_data[0][1].numpy()]

        partitions[cid] = tf.data.Dataset.from_tensor_slices(
            (np.stack(shard_x), np.array(shard_y))
        )
    return partitions


# ====================================================================== #
#  3.  TF LITE CONVERSION                                                 #
# ====================================================================== #

def convert_to_tflite(
    model: tf.keras.Model,
    output_path: str,
    quantise: bool = False,
) -> str:
    """
    Convert a Keras model to TensorFlow Lite format.

    Parameters
    ----------
    model : tf.keras.Model
    output_path : str
    quantise : bool
        If ``True``, apply dynamic-range (post-training) quantisation.

    Returns
    -------
    output_path : str
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantise:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(
        "TF Lite model saved → %s  (%.2f MB, quantised=%s)",
        output_path, size_mb, quantise,
    )
    return output_path


# ====================================================================== #
#  4.  FEDERATED LEARNING CYCLE  (main orchestrator)                      #
# ====================================================================== #

class FederatedLearningCycle:
    """
    End-to-end federated learning cycle orchestrating Parts 1–5.

    Parameters
    ----------
    config : FLCycleConfig
        The full cycle configuration.
    """

    def __init__(self, config: Optional[FLCycleConfig] = None) -> None:
        self.config = config or FLCycleConfig()
        self.global_model: Optional[tf.keras.Model] = None
        self.clients: List[FederatedClient] = []
        self.reputation_ledger: Optional[ClientReputationLedger] = None
        self.basic_ledger: Optional[ReputationLedger] = None
        self.selector: Optional[EnhancedClientSelector] = None
        self.validator: Optional[UpdateValidator] = None
        self.evaluator: Optional[FederatedModelEvaluator] = None

        # Round history
        self.history: Dict[str, list] = {
            "round": [],
            "global_accuracy": [],
            "selected_clients": [],
            "num_accepted": [],
            "num_rejected": [],
            "distillation_loss": [],
        }

    # ------------------------------------------------------------------ #
    #  Initialisation helpers                                              #
    # ------------------------------------------------------------------ #

    def load_global_model(self) -> tf.keras.Model:
        """Load the pre-trained EfficientNet model from disk."""
        cfg = self.config
        logger.info("Loading global model from %s …", cfg.model_path)
        # Register EfficientNet preprocessing so Keras can deserialize the .h5
        from tensorflow.keras.applications.efficientnet import (
            preprocess_input as _effnet_preprocess,
        )
        _custom = {"preprocess_input": _effnet_preprocess}
        model = tf.keras.models.load_model(
            cfg.model_path, custom_objects=_custom, compile=False,
        )
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
        """
        Instantiate ``FederatedClient`` objects for every device.

        Parameters
        ----------
        client_data : dict[str, tf.data.Dataset]
            Pre-partitioned local data per client.
        """
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
        return clients

    def setup_components(self) -> None:
        """
        Wire all the modules together: reputation ledger, client
        selector, update validator, and evaluator.
        """
        cfg = self.config
        assert self.global_model is not None, "Call load_global_model() first."
        assert len(self.clients) > 0, "Call create_clients() first."

        # -- Part 4: Reputation ledger --------------------------------- #
        self.reputation_ledger = ClientReputationLedger(config=cfg.reputation_config)
        for c in self.clients:
            self.reputation_ledger.register(c.client_id)
        # Create a basic ledger view for Part 1
        self.basic_ledger = self.reputation_ledger.as_basic_ledger()

        # -- Part 1: Enhanced client selector -------------------------- #
        self.selector = EnhancedClientSelector(
            clients=self.clients,
            reputation_ledger=self.basic_ledger,
            weights=cfg.selection_weights,
            target_k=cfg.clients_per_round,
        )

        # -- Part 2: Update validator ---------------------------------- #
        self.validator = UpdateValidator(
            global_model=self.global_model,
            reputation_ledger=self.basic_ledger,
            weights=cfg.contribution_weights,
            clipping=cfg.clipping_config,
            harmful_threshold=cfg.harmful_threshold,
            batch_size=cfg.local_batch_size,
        )

        # -- Part 5: Evaluator ---------------------------------------- #
        self.evaluator = FederatedModelEvaluator(
            model=self.global_model,
            model_name="effnet_global",
            reports_dir=cfg.reports_dir,
        )

        logger.info("All FL-cycle components initialised.")

    # ------------------------------------------------------------------ #
    #  Local training                                                     #
    # ------------------------------------------------------------------ #

    def _local_train(
        self,
        client: FederatedClient,
        global_weights: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], int]:
        """
        Train a local model replica for ``local_epochs`` on the client's
        private data, starting from the current global weights.

        Returns
        -------
        updated_weights : list[np.ndarray]
        num_samples : int
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
            logger.warning(
                "Client %s has no local data — returning global weights.",
                client.client_id,
            )
            return global_weights, 0

        dataset = client.local_data.batch(cfg.local_batch_size)
        local_model.fit(dataset, epochs=cfg.local_epochs, verbose=0)
        return local_model.get_weights(), client.metrics.data_volume

    # ------------------------------------------------------------------ #
    #  Sync reputation bridge                                             #
    # ------------------------------------------------------------------ #

    def _sync_reputation_to_basic_ledger(self) -> None:
        """
        Copy the extended ledger scores into the basic ``ReputationLedger``
        view used by Parts 1 & 2.
        """
        updated_basic = self.reputation_ledger.as_basic_ledger()
        # Update the internal _scores dict of the existing basic ledger
        # so the selector and validator see fresh reputations.
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
        Execute one complete federated round.

        Pipeline
        --------
        1.  **Client selection**   (Part 1)
        2.  **Local training**     — each selected client trains for
            ``local_epochs`` on its private data.
        3.  **Update validation & weighted aggregation**  (Part 2)
        4.  **Knowledge distillation**  (Part 3, optional)
        5.  **Reputation ledger update**  (Part 4)
        6.  Quick accuracy check on the server validation set.

        Parameters
        ----------
        current_round : int
        server_val_data : tf.data.Dataset
            Held-out server validation set  ``(x, y)``  for baseline
            comparison and post-round evaluation.
        proxy_data : tf.data.Dataset | None
            Unlabelled proxy inputs for distillation  (Part 3).
        supervised_data : tf.data.Dataset | None
            Labelled data for the combined distillation loss  (Part 3).

        Returns
        -------
        round_info : dict
        """
        cfg = self.config

        logger.info("=" * 70)
        logger.info("  ROUND %d / %d", current_round, cfg.global_rounds)
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

        global_weights = self.global_model.get_weights()

        # ── 2. Local training ──────────────────────────────────────── #
        client_updates: Dict[str, List[np.ndarray]] = {}
        data_volumes: Dict[str, int] = {}

        for client in selected:
            t0 = time.time()
            updated_w, n_samples = self._local_train(client, global_weights)
            elapsed = time.time() - t0

            client_updates[client.client_id] = updated_w
            data_volumes[client.client_id] = max(n_samples, 1)

            # Refresh client metrics
            client.metrics.last_selected_round = current_round
            client.metrics.inference_latency = elapsed / max(n_samples, 1)

            logger.debug(
                "Client %s trained — %d samples, %.2fs",
                client.client_id, n_samples, elapsed,
            )

        # ── 3. Update validation & aggregation  (Part 2) ──────────── #
        records: List[ClientUpdateRecord] = self.validator.validate_updates(
            client_updates=client_updates,
            data_volumes=data_volumes,
            server_val_data=server_val_data,
        )

        new_weights = self.validator.aggregate_weighted(
            records, global_weights,
        )
        self.global_model.set_weights(new_weights)
        self.validator.global_model.set_weights(new_weights)

        # Rejection statistics
        num_accepted = sum(1 for r in records if not r.rejected)
        num_rejected = sum(1 for r in records if r.rejected)
        logger.info(
            "Updates: %d accepted, %d rejected, out of %d total.",
            num_accepted, num_rejected, len(records),
        )

        # ── 4. Knowledge distillation  (Part 3, optional) ─────────── #
        distill_loss = None
        if cfg.enable_distillation and proxy_data is not None:
            # Only distill from clients with positive contribution
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
                # Update validator reference after distillation
                self.validator.global_model.set_weights(
                    self.global_model.get_weights()
                )
            else:
                logger.info(
                    "Skipping distillation — fewer than 2 contributing "
                    "clients (%d).", len(contribution_weights),
                )

        # ── 5. Reputation ledger update  (Part 4) ─────────────────── #
        updated_reps = update_ledger_from_records(
            ledger=self.reputation_ledger,
            records=records,
            current_round=current_round,
        )
        # Also feed reputation changes back via the basic (Part 1/2)
        # validator update_reputations for the simple ledger
        self.validator.update_reputations(records)

        # Sync extended ledger → basic ledger used by selector/validator
        self._sync_reputation_to_basic_ledger()

        logger.info(
            "Reputation update complete — top 5: %s",
            dict(list(self.reputation_ledger.ranked()[:5])),
        )

        # ── 6. Quick evaluation on val set ────────────────────────── #
        val_result = self.global_model.evaluate(
            server_val_data.batch(cfg.local_batch_size),
            verbose=0,
            return_dict=True,
        )
        acc = val_result.get("accuracy", 0.0)
        logger.info("Round %d — post-round accuracy: %.4f", current_round, acc)

        # ── Collect summary ────────────────────────────────────────── #
        round_info = {
            "round": current_round,
            "selected": selected_ids,
            "num_accepted": num_accepted,
            "num_rejected": num_rejected,
            "records": records,
            "global_accuracy": acc,
            "distillation_loss": distill_loss,
        }
        return round_info

    # ------------------------------------------------------------------ #
    #  Full FL cycle                                                      #
    # ------------------------------------------------------------------ #

    def run(
        self,
        server_val_data: tf.data.Dataset,
        test_data: tf.data.Dataset,
        proxy_data: Optional[tf.data.Dataset] = None,
        supervised_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, list]:
        """
        Run the full federated learning cycle for ``global_rounds``
        rounds.

        Parameters
        ----------
        server_val_data : tf.data.Dataset
            Held-out validation set used for update scoring & quick eval.
        test_data : tf.data.Dataset
            Independent test set for full evaluation (Part 5).
        proxy_data : tf.data.Dataset | None
            Unlabelled proxy data for knowledge distillation (Part 3).
        supervised_data : tf.data.Dataset | None
            Labelled data for the combined distillation loss (Part 3).

        Returns
        -------
        history : dict
        """
        cfg = self.config

        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║  FEDERATED LEARNING CYCLE — START                   ║")
        logger.info("║  Devices: %3d  |  Rounds: %3d  |  Local epochs: %d  ║",
                     cfg.num_devices, cfg.global_rounds, cfg.local_epochs)
        logger.info("╚══════════════════════════════════════════════════════╝")

        # -- Pre-cycle evaluation (baseline) --------------------------- #
        logger.info("Evaluating baseline model before federated training …")
        baseline_report = self.evaluator.evaluate(
            test_data=test_data,
            batch_size=cfg.local_batch_size,
            federated_round=0,
            extra_info={"stage": "baseline"},
        )
        self.evaluator.save_report(baseline_report, tag="round_000_baseline")
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
            self.history["global_accuracy"].append(info["global_accuracy"])
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
                logger.info("Running full evaluation (round %d) …", t)
                report = self.evaluator.evaluate(
                    test_data=test_data,
                    batch_size=cfg.local_batch_size,
                    federated_round=t,
                    extra_info={
                        "accepted": info["num_accepted"],
                        "rejected": info["num_rejected"],
                        "distillation_loss": info["distillation_loss"],
                    },
                )
                self.evaluator.save_report(report, tag=f"round_{t:03d}")
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

        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║  FEDERATED LEARNING CYCLE — COMPLETE                ║")
        logger.info("║  Total time: %.1fs                                  ║", total_elapsed)
        logger.info("╚══════════════════════════════════════════════════════╝")

        # -- Save comparison report ------------------------------------ #
        if len(all_reports) > 1:
            self.evaluator.save_comparison_report(all_reports)

        # -- Save reputation ledger ------------------------------------ #
        ledger_path = Path(cfg.reports_dir) / "reputation_ledger_final.json"
        self.reputation_ledger.save(str(ledger_path))

        # -- Convert to TF Lite ---------------------------------------- #
        convert_to_tflite(
            self.global_model,
            cfg.tflite_output_path,
            quantise=False,
        )
        # Also save a quantised version
        convert_to_tflite(
            self.global_model,
            cfg.tflite_output_path.replace(".tflite", "_quantised.tflite"),
            quantise=True,
        )

        # -- Print final summary --------------------------------------- #
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

        best_idx = int(np.argmax(h["global_accuracy"]))
        best_round = h["round"][best_idx]
        best_acc = h["global_accuracy"][best_idx]
        final_acc = h["global_accuracy"][-1]

        # Reputation statistics
        stats = self.reputation_ledger.statistics()

        print("\n" + "=" * 60)
        print("  FEDERATED LEARNING CYCLE — FINAL SUMMARY")
        print("=" * 60)
        print(f"  Total rounds:          {len(h['round'])}")
        print(f"  Best accuracy:         {best_acc:.4f}  (round {best_round})")
        print(f"  Final accuracy:        {final_acc:.4f}")
        print(f"  Mean accuracy:         {np.mean(h['global_accuracy']):.4f}")
        print(f"  Total accepted:        {sum(h['num_accepted'])}")
        print(f"  Total rejected:        {sum(h['num_rejected'])}")
        print(f"  Reputation — mean:     {stats.get('mean_reputation', 0):.4f}")
        print(f"  Reputation — std:      {stats.get('std_reputation', 0):.4f}")

        kd_losses = [l for l in h["distillation_loss"] if l is not None]
        if kd_losses:
            print(f"  Avg distillation loss: {np.mean(kd_losses):.5f}")
        print(f"  TF Lite model saved:   {self.config.tflite_output_path}")
        print("=" * 60 + "\n")


# ====================================================================== #
#  5.  ENTRY POINT                                                        #
# ====================================================================== #

def main() -> None:
    """
    Main entry point — runs the full Enhanced Federated Learning Cycle
    for DeepFake detection.
    """
    np.random.seed(42)
    tf.random.set_seed(42)

    # ------------------------------------------------------------------ #
    #  Configuration                                                      #
    # ------------------------------------------------------------------ #
    config = FLCycleConfig(
        model_path="effnet_ffpp_small_data.h5",
        num_devices=100,
        local_epochs=5,
        global_rounds=50,
        clients_per_round=15,
        local_batch_size=32,
        local_lr=1e-4,
        eval_every=10,
        enable_distillation=True,
    )

    cycle = FederatedLearningCycle(config)

    # ------------------------------------------------------------------ #
    #  Load global model                                                  #
    # ------------------------------------------------------------------ #
    model = cycle.load_global_model()
    input_shape = model.input_shape[1:]        # strip batch dim
    config.input_shape = input_shape
    logger.info("Model input shape: %s", input_shape)

    # ------------------------------------------------------------------ #
    #  Prepare data  (synthetic — replace with real FF++ c23 loaders)     #
    # ------------------------------------------------------------------ #
    #
    #  In production, replace these synthetic generators with actual
    #  FF++ c23 data loaders:
    #
    #    train_data = load_ffpp_c23_train(...)      # for client partitions
    #    val_data   = load_ffpp_c23_val(...)        # server validation
    #    test_data  = load_ffpp_c23_test(...)       # independent test set
    #    proxy_data = load_ffpp_c23_unlabelled(...) # for distillation
    #
    logger.info("Generating synthetic data for %d clients …", config.num_devices)

    TOTAL_TRAIN_SAMPLES = config.num_devices * 10   # 10 samples/client
    VAL_SAMPLES   = 200
    TEST_SAMPLES  = 300
    PROXY_SAMPLES = 150

    train_data = _generate_synthetic_data(
        TOTAL_TRAIN_SAMPLES, input_shape, seed=1,
    )
    server_val_data = _generate_synthetic_data(
        VAL_SAMPLES, input_shape, seed=2,
    )
    test_data = _generate_synthetic_data(
        TEST_SAMPLES, input_shape, seed=3,
    )
    proxy_data = _generate_proxy_data(
        PROXY_SAMPLES, input_shape, seed=4,
    )
    supervised_data = _generate_synthetic_data(
        100, input_shape, seed=5,
    )

    # Partition training data across clients (IID)
    client_data = partition_data_iid(
        train_data, config.num_devices, seed=42,
    )

    # ------------------------------------------------------------------ #
    #  Create clients & wire components                                   #
    # ------------------------------------------------------------------ #
    cycle.create_clients(client_data)
    cycle.setup_components()

    # ------------------------------------------------------------------ #
    #  Run the Federated Learning Cycle                                   #
    # ------------------------------------------------------------------ #
    history = cycle.run(
        server_val_data=server_val_data,
        test_data=test_data,
        proxy_data=proxy_data,
        supervised_data=supervised_data,
    )

    logger.info("Federated Learning Cycle finished. History keys: %s",
                list(history.keys()))


# ====================================================================== #
#  DEMO / SMOKE-TEST  (lightweight — reduced params for quick run)        #
# ====================================================================== #

if __name__ == "__main__":
    print("\n===  Enhanced Federated Learning Cycle — Demo  ===\n")

    np.random.seed(42)
    tf.random.set_seed(42)

    # ---- Use a small synthetic model for smoke-testing --------------- #
    # (The real cycle uses effnet_ffpp_small_data.h5 via main())
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
    print(f"Demo model — {demo_model.count_params()} params, "
          f"input shape {demo_model.input_shape}")

    # ---- Configuration (reduced for speed) --------------------------- #
    demo_config = FLCycleConfig(
        model_path="(in-memory demo)",
        num_devices=8,
        local_epochs=1,
        global_rounds=3,
        clients_per_round=4,
        local_batch_size=16,
        local_lr=1e-3,
        eval_every=1,
        enable_distillation=True,
        distillation_config=DistillationConfig(
            temperature=3.0,
            lam=0.7,
            epochs=2,
            batch_size=16,
            learning_rate=1e-3,
        ),
    )

    # ---- Synthetic data ---------------------------------------------- #
    N_CLIENT = demo_config.num_devices
    SAMPLES_PER_CLIENT = 30
    TOTAL = N_CLIENT * SAMPLES_PER_CLIENT
    input_shape = (INPUT_DIM,)

    train_ds = _generate_synthetic_data(TOTAL, input_shape, seed=10)
    val_ds   = _generate_synthetic_data(100, input_shape, seed=20)
    test_ds  = _generate_synthetic_data(120, input_shape, seed=30)
    proxy_ds = _generate_proxy_data(80, input_shape, seed=40)
    sup_ds   = _generate_synthetic_data(60, input_shape, seed=50)

    client_data = partition_data_iid(train_ds, N_CLIENT, seed=42)

    # ---- Build cycle ------------------------------------------------- #
    cycle = FederatedLearningCycle(demo_config)
    cycle.global_model = demo_model         # skip load_model for demo
    cycle.create_clients(client_data)
    cycle.setup_components()

    # ---- Run --------------------------------------------------------- #
    history = cycle.run(
        server_val_data=val_ds,
        test_data=test_ds,
        proxy_data=proxy_ds,
        supervised_data=sup_ds,
    )

    # ---- Show history ------------------------------------------------ #
    print("\nRound-by-round accuracy:")
    for rnd, acc in zip(history["round"], history["global_accuracy"]):
        print(f"  Round {rnd}: {acc:.4f}")

    # ---- TF Lite check ----------------------------------------------- #
    tflite_path = demo_config.tflite_output_path
    if Path(tflite_path).exists():
        size_kb = Path(tflite_path).stat().st_size / 1024
        print(f"\nTF Lite model: {tflite_path} ({size_kb:.1f} KB)")
        # Clean up demo artifacts
        Path(tflite_path).unlink(missing_ok=True)
        q_path = tflite_path.replace(".tflite", "_quantised.tflite")
        Path(q_path).unlink(missing_ok=True)

    print("\nDone.")
