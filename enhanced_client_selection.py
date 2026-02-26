"""
Enhanced Federated Client Selection Strategy
=============================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

Implements a multi-criteria client selection mechanism for each federated
round, scoring clients on:
  - Local validation performance  (V)
  - Data volume                   (D)
  - Inference latency             (L)  — penalises slow clients
  - Reputation                    (R)  — maintained across rounds
  - Staleness                     (S)  — penalises long-absent clients

Reference pseudo-code from the thesis proposal is faithfully reproduced.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ====================================================================== #
#  1.  DATA STRUCTURES                                                    #
# ====================================================================== #

@dataclass
class ClientMetrics:
    """Raw metric snapshot collected from / about a single client."""
    local_validation_metric: float = 0.0   # e.g. local F1 on holdout split
    data_volume: int = 0                    # number of local samples
    inference_latency: float = 0.0          # seconds for a forward pass batch
    last_selected_round: int = 0            # last round this client participated


@dataclass
class FederatedClient:
    """
    Represents one federated learning participant.

    Parameters
    ----------
    client_id : str
        Unique human-readable identifier.
    local_data : tf.data.Dataset | None
        The client's private dataset (images + labels).
    metrics : ClientMetrics
        Latest raw metrics for this client.
    """
    client_id: str
    local_data: Optional[tf.data.Dataset] = None
    metrics: ClientMetrics = field(default_factory=ClientMetrics)

    def __repr__(self) -> str:
        return (
            f"FederatedClient(id={self.client_id!r}, "
            f"samples={self.metrics.data_volume})"
        )


# ====================================================================== #
#  2.  REPUTATION LEDGER                                                  #
# ====================================================================== #

class ReputationLedger:
    """
    Tracks a per-client reputation score in [0, 1].

    Reputation increases when the client's model update *improves* the
    global validation metric and decreases (or stays flat) otherwise.
    An exponential moving average (EMA) keeps the score smooth.

    Parameters
    ----------
    default_reputation : float
        Initial reputation assigned to every newly registered client.
    ema_alpha : float
        Smoothing factor for the EMA update  (higher → faster adaptation).
    reward : float
        Reputation bump for a *good* update.
    penalty : float
        Reputation decrease for a *bad* or *no-op* update.
    """

    def __init__(
        self,
        default_reputation: float = 0.5,
        ema_alpha: float = 0.3,
        reward: float = 0.1,
        penalty: float = 0.05,
    ) -> None:
        self.default_reputation = default_reputation
        self.ema_alpha = ema_alpha
        self.reward = reward
        self.penalty = penalty
        self._scores: Dict[str, float] = {}

    # ---- public API -------------------------------------------------- #

    def register(self, client_id: str) -> None:
        """Register a client with the default reputation."""
        if client_id not in self._scores:
            self._scores[client_id] = self.default_reputation

    def get(self, client_id: str) -> float:
        """Return the current reputation for *client_id*."""
        return self._scores.get(client_id, self.default_reputation)

    def update(self, client_id: str, update_was_beneficial: bool) -> None:
        """
        Update the reputation of *client_id* after a federated round.

        Parameters
        ----------
        client_id : str
        update_was_beneficial : bool
            ``True`` if the client's local update improved (or at least did
            not degrade) the global model on a held-out validation set.
        """
        old = self.get(client_id)
        delta = self.reward if update_was_beneficial else -self.penalty
        new = old + self.ema_alpha * delta
        self._scores[client_id] = float(np.clip(new, 0.0, 1.0))
        logger.debug(
            "Reputation %s: %.4f → %.4f (beneficial=%s)",
            client_id, old, self._scores[client_id], update_was_beneficial,
        )

    def summary(self) -> Dict[str, float]:
        """Return a copy of the full reputation table."""
        return dict(self._scores)


# ====================================================================== #
#  3.  NORMALISATION & SCORING HELPERS                                    #
# ====================================================================== #

def _min_max_normalise(values: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1].  Returns zeros when range = 0."""
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lo) / (hi - lo)


def _log_scale(values: np.ndarray) -> np.ndarray:
    """Log-scale data volumes before normalising (handles zero gracefully)."""
    return np.log1p(values.astype(np.float64))


def staleness_penalty(last_selected_round: int, current_round: int) -> float:
    """
    Compute a staleness penalty in [0, 1].

    Uses an exponential decay:  ``1 - exp(-gap / 5)``  so that a client
    absent for ~15 rounds is nearly fully penalised.
    """
    gap = max(current_round - last_selected_round, 0)
    return 1.0 - math.exp(-gap / 5.0)


# ====================================================================== #
#  4.  ENHANCED CLIENT SELECTION STRATEGY                                 #
# ====================================================================== #

@dataclass
class SelectionWeights:
    """Tuneable weights for the multi-criteria scoring function."""
    w_v: float = 0.30   # local validation performance
    w_d: float = 0.20   # data volume
    w_l: float = 0.15   # latency  (applied to 1 - L_i)
    w_r: float = 0.25   # reputation
    w_s: float = 0.10   # staleness penalty (subtracted)

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.w_v, self.w_d, self.w_l, self.w_r, self.w_s)


class EnhancedClientSelector:
    """
    Implements the *Enhanced Federated Client Selection Strategy*.

    Per round the selector:
      1. Collects raw metrics from every candidate client.
      2. Normalises them across the pool.
      3. Computes a composite score per client.
      4. Returns the top-K (or above-threshold) clients.

    Parameters
    ----------
    clients : list[FederatedClient]
        The full pool of federated clients.
    reputation_ledger : ReputationLedger
        Shared ledger that persists across rounds.
    weights : SelectionWeights
        Relative importance of each scoring dimension.
    target_k : int
        Number of clients to select each round (top-K mode).
    threshold : float | None
        If set, *all* clients with ``score >= threshold`` are selected
        instead of a fixed K.
    """

    def __init__(
        self,
        clients: List[FederatedClient],
        reputation_ledger: ReputationLedger,
        weights: Optional[SelectionWeights] = None,
        target_k: int = 5,
        threshold: Optional[float] = None,
    ) -> None:
        self.clients = {c.client_id: c for c in clients}
        self.ledger = reputation_ledger
        self.weights = weights or SelectionWeights()
        self.target_k = target_k
        self.threshold = threshold

        # Register every client in the ledger
        for cid in self.clients:
            self.ledger.register(cid)

    # ------------------------------------------------------------------ #
    #  Core selection logic                                               #
    # ------------------------------------------------------------------ #

    def score_clients(
        self, current_round: int
    ) -> List[Tuple[str, float]]:
        """
        Compute and return ``(client_id, score)`` for every client,
        sorted descending by score.
        """
        ids = list(self.clients.keys())
        n = len(ids)

        # --- gather raw vectors ---------------------------------------- #
        raw_v = np.array([self.clients[i].metrics.local_validation_metric for i in ids])
        raw_d = np.array([self.clients[i].metrics.data_volume for i in ids])
        raw_l = np.array([self.clients[i].metrics.inference_latency for i in ids])

        # --- normalise ------------------------------------------------- #
        V = _min_max_normalise(raw_v)                    # higher is better
        D = _min_max_normalise(_log_scale(raw_d))        # log-scaled, higher is better
        L = _min_max_normalise(raw_l)                    # 1 = slowest (penalty)
        R = np.array([self.ledger.get(i) for i in ids])  # already in [0,1]
        S = np.array([
            staleness_penalty(self.clients[i].metrics.last_selected_round, current_round)
            for i in ids
        ])

        # --- composite score ------------------------------------------- #
        w = self.weights
        scores = (
            w.w_v * V
            + w.w_d * D
            + w.w_l * (1.0 - L)
            + w.w_r * R
            - w.w_s * S
        )

        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
        return ranked

    def select(
        self, current_round: int
    ) -> List[FederatedClient]:
        """
        Select clients for the current federated round.

        Returns
        -------
        list[FederatedClient]
            The chosen participants (top-K or above-threshold).
        """
        ranked = self.score_clients(current_round)

        if self.threshold is not None:
            selected_ids = [cid for cid, sc in ranked if sc >= self.threshold]
            # Guarantee at least 1 client even when none meets threshold
            if not selected_ids:
                selected_ids = [ranked[0][0]]
                logger.warning(
                    "Round %d — no client met threshold %.3f; "
                    "falling back to best client %s (score %.4f).",
                    current_round, self.threshold,
                    ranked[0][0], ranked[0][1],
                )
        else:
            k = min(self.target_k, len(ranked))
            selected_ids = [cid for cid, _ in ranked[:k]]

        logger.info(
            "Round %d — selected %d / %d clients: %s",
            current_round, len(selected_ids), len(ranked), selected_ids,
        )
        return [self.clients[cid] for cid in selected_ids]


# ====================================================================== #
#  5.  FEDERATED ROUND RUNNER  (skeleton compatible with TFF)             #
# ====================================================================== #

class FederatedRoundRunner:
    """
    Orchestrates the full Enhanced Federated Learning cycle.

    This class ties together:
      * The global model (loaded from the .h5 file)
      * The enhanced client selector
      * Local training on selected clients
      * Federated averaging of model updates
      * Reputation ledger updates after each round

    Parameters
    ----------
    global_model_path : str
        Path to the initial global Keras model (``*.h5``).
    clients : list[FederatedClient]
        All available federated clients.
    selector : EnhancedClientSelector
        The client selection strategy instance.
    local_epochs : int
        Number of local training epochs per client per round.
    local_batch_size : int
        Batch size for local training.
    learning_rate : float
        Learning rate for local SGD.
    """

    def __init__(
        self,
        global_model_path: str,
        clients: List[FederatedClient],
        selector: EnhancedClientSelector,
        local_epochs: int = 1,
        local_batch_size: int = 32,
        learning_rate: float = 1e-4,
    ) -> None:
        # Register EfficientNet preprocessing so Keras can deserialize the .h5
        from tensorflow.keras.applications.efficientnet import (
            preprocess_input as _effnet_preprocess,
        )
        _custom = {"preprocess_input": _effnet_preprocess}
        self.global_model: tf.keras.Model = tf.keras.models.load_model(
            global_model_path, custom_objects=_custom, compile=False,
        )
        self.global_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.clients = clients
        self.selector = selector
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.learning_rate = learning_rate

    # ------------------------------------------------------------------ #
    #  Federated Averaging helpers                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fedavg(
        global_weights: List[np.ndarray],
        client_weights_list: List[List[np.ndarray]],
        sample_counts: List[int],
    ) -> List[np.ndarray]:
        """
        Weighted Federated Averaging (FedAvg).

        Each client's contribution is proportional to its local sample count.
        """
        total = sum(sample_counts)
        averaged = [
            np.zeros_like(w) for w in global_weights
        ]
        for cw, n in zip(client_weights_list, sample_counts):
            for idx, w in enumerate(cw):
                averaged[idx] += w * (n / total)
        return averaged

    # ------------------------------------------------------------------ #
    #  Local training on a single client                                  #
    # ------------------------------------------------------------------ #

    def _local_train(
        self,
        client: FederatedClient,
        global_weights: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], int, float]:
        """
        Perform local training on *client* starting from *global_weights*.

        Returns
        -------
        updated_weights : list[np.ndarray]
        num_samples : int
        local_val_metric : float   (e.g. accuracy on local holdout)
        """
        # Build a fresh copy with global weights
        local_model: tf.keras.Model = tf.keras.models.clone_model(self.global_model)
        local_model.build(self.global_model.input_shape)
        local_model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        local_model.set_weights(global_weights)

        if client.local_data is None:
            logger.warning("Client %s has no local data — skipping.", client.client_id)
            return global_weights, 0, 0.0

        dataset = client.local_data.batch(self.local_batch_size)

        # Measure inference latency (single batch)
        t0 = time.perf_counter()
        for batch in dataset.take(1):
            local_model.predict(batch[0], verbose=0)
        client.metrics.inference_latency = time.perf_counter() - t0

        # Local training
        local_model.fit(dataset, epochs=self.local_epochs, verbose=0)

        # Evaluate on same data as a proxy (in production: use a held-out split)
        results = local_model.evaluate(dataset, verbose=0, return_dict=True)
        local_val = results.get("accuracy", 0.0)

        num_samples = client.metrics.data_volume

        return local_model.get_weights(), num_samples, local_val

    # ------------------------------------------------------------------ #
    #  Validate an individual client update against the global model      #
    # ------------------------------------------------------------------ #

    def _validate_update(
        self,
        updated_weights: List[np.ndarray],
        validation_data: Optional[tf.data.Dataset],
    ) -> Optional[float]:
        """
        Evaluate *updated_weights* on a global validation set.

        Returns the accuracy (or ``None`` if no validation data is provided).
        """
        if validation_data is None:
            return None
        temp_model: tf.keras.Model = tf.keras.models.clone_model(self.global_model)
        temp_model.build(self.global_model.input_shape)
        temp_model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        temp_model.set_weights(updated_weights)
        results = temp_model.evaluate(
            validation_data.batch(self.local_batch_size), verbose=0, return_dict=True,
        )
        return results.get("accuracy", 0.0)

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        num_rounds: int = 10,
        global_val_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, list]:
        """
        Execute the full federated learning cycle.

        Parameters
        ----------
        num_rounds : int
            Total communication rounds ``T``.
        global_val_data : tf.data.Dataset | None
            A small global validation set used to judge update quality
            (for the reputation ledger).

        Returns
        -------
        history : dict
            ``{"round": [...], "global_acc": [...], "selected_clients": [...]}``
        """
        history: Dict[str, list] = {
            "round": [],
            "global_accuracy": [],
            "selected_clients": [],
        }

        # Compute baseline accuracy before federated training
        if global_val_data is not None:
            baseline = self.global_model.evaluate(
                global_val_data.batch(self.local_batch_size),
                verbose=0,
                return_dict=True,
            )
            logger.info("Baseline global accuracy: %.4f", baseline.get("accuracy", 0.0))

        for t in range(1, num_rounds + 1):
            logger.info("=" * 60)
            logger.info("ROUND %d / %d", t, num_rounds)
            logger.info("=" * 60)

            # ---- 1. Select clients ----------------------------------- #
            selected = self.selector.select(current_round=t)

            global_weights = self.global_model.get_weights()
            client_updates: List[List[np.ndarray]] = []
            sample_counts: List[int] = []

            # ---- 2. Local training ----------------------------------- #
            for client in selected:
                updated_w, n_samples, local_val = self._local_train(
                    client, global_weights
                )
                client.metrics.local_validation_metric = local_val
                client.metrics.last_selected_round = t

                client_updates.append(updated_w)
                sample_counts.append(max(n_samples, 1))  # avoid div-by-zero

                # ---- 3. Reputation update ----------------------------- #
                if global_val_data is not None:
                    update_acc = self._validate_update(updated_w, global_val_data)
                    current_acc = self._validate_update(global_weights, global_val_data)
                    beneficial = (
                        update_acc is not None
                        and current_acc is not None
                        and update_acc >= current_acc - 1e-4
                    )
                else:
                    # Optimistic: assume beneficial when we cannot verify
                    beneficial = True

                self.selector.ledger.update(client.client_id, beneficial)

            # ---- 4. Federated averaging ------------------------------ #
            if client_updates:
                new_global = self._fedavg(global_weights, client_updates, sample_counts)
                self.global_model.set_weights(new_global)

            # ---- 5. Global evaluation -------------------------------- #
            round_acc = None
            if global_val_data is not None:
                result = self.global_model.evaluate(
                    global_val_data.batch(self.local_batch_size),
                    verbose=0,
                    return_dict=True,
                )
                round_acc = result.get("accuracy", 0.0)
                logger.info("Round %d global accuracy: %.4f", t, round_acc)

            history["round"].append(t)
            history["global_accuracy"].append(round_acc)
            history["selected_clients"].append([c.client_id for c in selected])

        logger.info("Federated training complete — %d rounds.", num_rounds)
        return history


# ====================================================================== #
#  6.  CONVENIENCE FACTORY                                                #
# ====================================================================== #

def build_default_pipeline(
    model_path: str = "effnet_ffpp_small_data.h5",
    num_clients: int = 10,
    target_k: int = 5,
    threshold: Optional[float] = None,
    weights: Optional[SelectionWeights] = None,
) -> Tuple[FederatedRoundRunner, List[FederatedClient]]:
    """
    Quick-start helper that wires up all components.

    In practice you would replace the synthetic client list with real
    data partitions.

    Returns
    -------
    runner : FederatedRoundRunner
    clients : list[FederatedClient]
    """
    # --- Create placeholder clients (replace with real data loaders) --- #
    clients: List[FederatedClient] = []
    for i in range(num_clients):
        c = FederatedClient(
            client_id=f"client_{i:02d}",
            local_data=None,           # <-- plug real tf.data.Dataset here
            metrics=ClientMetrics(
                local_validation_metric=np.random.uniform(0.5, 0.95),
                data_volume=int(np.random.randint(200, 5000)),
                inference_latency=np.random.uniform(0.01, 0.5),
                last_selected_round=0,
            ),
        )
        clients.append(c)

    ledger = ReputationLedger()
    selector = EnhancedClientSelector(
        clients=clients,
        reputation_ledger=ledger,
        weights=weights or SelectionWeights(),
        target_k=target_k,
        threshold=threshold,
    )
    runner = FederatedRoundRunner(
        global_model_path=model_path,
        clients=clients,
        selector=selector,
    )
    return runner, clients


# ====================================================================== #
#  7.  MAIN — demo / smoke test                                           #
# ====================================================================== #

if __name__ == "__main__":
    # -------------------------------------------------------------- #
    #  Standalone demo:  runs selection scoring with synthetic clients #
    #  (no real data or model loading — safe to execute anywhere).    #
    # -------------------------------------------------------------- #

    NUM_CLIENTS = 10
    TARGET_K = 4
    NUM_ROUNDS = 5

    # Create synthetic client pool
    np.random.seed(42)
    demo_clients: List[FederatedClient] = []
    for idx in range(NUM_CLIENTS):
        demo_clients.append(
            FederatedClient(
                client_id=f"client_{idx:02d}",
                local_data=None,
                metrics=ClientMetrics(
                    local_validation_metric=np.random.uniform(0.4, 0.95),
                    data_volume=int(np.random.randint(100, 8000)),
                    inference_latency=np.random.uniform(0.01, 1.0),
                    last_selected_round=0,
                ),
            )
        )

    ledger = ReputationLedger()
    selector = EnhancedClientSelector(
        clients=demo_clients,
        reputation_ledger=ledger,
        weights=SelectionWeights(w_v=0.30, w_d=0.20, w_l=0.15, w_r=0.25, w_s=0.10),
        target_k=TARGET_K,
    )

    print("\n===  Enhanced Client Selection — Demo  ===\n")
    for rnd in range(1, NUM_ROUNDS + 1):
        ranked = selector.score_clients(current_round=rnd)
        selected = selector.select(current_round=rnd)

        print(f"\n--- Round {rnd} ---")
        print(f"{'Client':<14} {'Score':>8}")
        print("-" * 24)
        for cid, score in ranked:
            marker = " ✓" if cid in {c.client_id for c in selected} else ""
            print(f"{cid:<14} {score:>8.4f}{marker}")

        # Simulate reputation changes (random for demo)
        for c in selected:
            beneficial = np.random.random() > 0.3   # 70 % chance of good update
            ledger.update(c.client_id, bool(beneficial))
            c.metrics.last_selected_round = rnd

    print("\n--- Final Reputation Ledger ---")
    for cid, rep in sorted(ledger.summary().items()):
        print(f"  {cid}: {rep:.4f}")
