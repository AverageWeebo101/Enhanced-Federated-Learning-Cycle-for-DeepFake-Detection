"""
Update Validation and Contribution Weighing
============================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

After each federated round, every client update is individually validated
and assigned a **contribution weight** before aggregation.  The pipeline:

  1. Norm check — flag / clip suspiciously large updates.
  2. Server-side validation gain — apply the update to a temp copy of the
     global model and measure the score delta on a held-out server set.
  3. Similarity check — cosine similarity with the recent global update
     history (catches free-riders that echo old gradients).
  4. Multi-criteria raw contribution score.
  5. Weighted aggregation (contribution-weighted FedAvg).
  6. Reputation ledger feedback from observed gains.

Imports the shared data-structures from ``enhanced_client_selection.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ---------- shared types from Part 1 ---------------------------------- #
from enhanced_client_selection import (
    FederatedClient,
    ClientMetrics,
    ReputationLedger,
    _min_max_normalise,
    _log_scale,
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
class ContributionWeights:
    """
    Tuneable weights for the contribution scoring formula.

    ``raw = α·G_i + β·sim_i + γ·norm(D_i) + δ·R_i``
    """
    alpha: float = 0.40   # server-side validation gain
    beta:  float = 0.15   # cosine similarity to global update history
    gamma: float = 0.20   # normalised data volume
    delta: float = 0.25   # reputation

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.alpha, self.beta, self.gamma, self.delta)


@dataclass
class ClippingConfig:
    """Parameters for the norm-based update clipping / rejection."""
    clip_threshold: float = 10.0     # max allowed L2 norm of a flattened update
    clip_value: Optional[float] = None  # if set, clip *to* this norm instead of rejecting


# ====================================================================== #
#  2.  HELPER UTILITIES                                                   #
# ====================================================================== #

def flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    """Concatenate a list of weight arrays into a single 1-D vector."""
    return np.concatenate([w.ravel() for w in weights])


def unflatten_weights(
    flat: np.ndarray,
    shapes: List[Tuple[int, ...]],
) -> List[np.ndarray]:
    """Inverse of ``flatten_weights``: split a 1-D vector back into arrays."""
    arrays: List[np.ndarray] = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        arrays.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return arrays


def compute_update_delta(
    global_weights: List[np.ndarray],
    updated_weights: List[np.ndarray],
) -> List[np.ndarray]:
    """Return the element-wise difference  ``updated − global``."""
    return [u - g for u, g in zip(updated_weights, global_weights)]


def apply_update(
    base_weights: List[np.ndarray],
    delta: List[np.ndarray],
    scale: float = 1.0,
) -> List[np.ndarray]:
    """Return ``base + scale * delta`` (per-layer)."""
    return [b + scale * d for b, d in zip(base_weights, delta)]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.

    Returns 0.0 when either vector has near-zero norm (avoids NaN).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _normalise_scalar_to_01(
    values: np.ndarray,
) -> np.ndarray:
    """Scale an array into [0, 1] via min-max.  Alias for readability."""
    return _min_max_normalise(values)


# ====================================================================== #
#  3.  GLOBAL UPDATE HISTORY                                              #
# ====================================================================== #

class GlobalUpdateHistory:
    """
    Maintains a rolling window of the last *N* aggregated global updates
    (as flattened vectors) so the validator can compute cosine similarity
    between a client update and "what the model has been learning lately".

    Parameters
    ----------
    max_history : int
        Maximum number of past global deltas to keep.
    """

    def __init__(self, max_history: int = 10) -> None:
        self.max_history = max_history
        self._history: List[np.ndarray] = []   # each entry is a 1-D vector

    def push(self, global_delta_flat: np.ndarray) -> None:
        """Append the latest aggregated global delta."""
        self._history.append(global_delta_flat.copy())
        if len(self._history) > self.max_history:
            self._history.pop(0)

    @property
    def mean_direction(self) -> Optional[np.ndarray]:
        """
        Return the mean direction of stored history.

        This single vector captures the *average trend* the global model
        has been moving in.  Returns ``None`` before the first round.
        """
        if not self._history:
            return None
        stacked = np.stack(self._history, axis=0)
        return stacked.mean(axis=0)

    @property
    def size(self) -> int:
        return len(self._history)


# ====================================================================== #
#  4.  UPDATE VALIDATOR & CONTRIBUTION SCORER                             #
# ====================================================================== #

@dataclass
class ClientUpdateRecord:
    """Result of validating a single client's update."""
    client_id: str
    delta: List[np.ndarray]          # raw weight delta  (updated − global)
    norm: float = 0.0                # L2 norm of flattened delta
    is_suspicious: bool = False      # flagged by norm check
    validation_gain: float = 0.0     # G_i = new_score − baseline_score
    similarity: float = 0.0         # cosine similarity with history
    raw_contribution: float = 0.0    # before normalisation
    contribution_weight: float = 0.0 # final c_i in [0, 1]
    rejected: bool = False           # update completely rejected


class UpdateValidator:
    """
    Validates client updates and computes contribution weights.

    This is the **core class** of the second part of the federated cycle.

    Parameters
    ----------
    global_model : tf.keras.Model
        The current global model (used to create temp copies for eval).
    reputation_ledger : ReputationLedger
        Shared reputation tracker (read for R_i, written after scoring).
    weights : ContributionWeights
        α, β, γ, δ for the composite score.
    clipping : ClippingConfig
        Norm-clipping parameters.
    harmful_threshold : float
        ε — if ``G_i < −ε`` the update is rejected outright.
    batch_size : int
        Batch size when evaluating on the server validation set.
    eval_metric : str
        Name of the Keras metric to use as *score* (e.g. ``"accuracy"``
        or ``"f1_score"``).
    """

    def __init__(
        self,
        global_model: tf.keras.Model,
        reputation_ledger: ReputationLedger,
        weights: Optional[ContributionWeights] = None,
        clipping: Optional[ClippingConfig] = None,
        harmful_threshold: float = 0.02,
        batch_size: int = 32,
        eval_metric: str = "accuracy",
    ) -> None:
        self.global_model = global_model
        self.ledger = reputation_ledger
        self.weights = weights or ContributionWeights()
        self.clipping = clipping or ClippingConfig()
        self.harmful_threshold = harmful_threshold
        self.batch_size = batch_size
        self.eval_metric = eval_metric
        self.update_history = GlobalUpdateHistory()

    # ------------------------------------------------------------------ #
    #  Evaluation helper                                                  #
    # ------------------------------------------------------------------ #

    def _evaluate(
        self,
        model_weights: List[np.ndarray],
        val_data: tf.data.Dataset,
    ) -> float:
        """
        Evaluate *model_weights* on *val_data* by creating a temporary
        model clone, setting the weights, and running ``evaluate()``.

        Returns the scalar value of ``self.eval_metric``.
        """
        temp: tf.keras.Model = tf.keras.models.clone_model(self.global_model)
        temp.build(self.global_model.input_shape)
        temp.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        temp.set_weights(model_weights)
        results = temp.evaluate(
            val_data.batch(self.batch_size), verbose=0, return_dict=True,
        )
        return float(results.get(self.eval_metric, 0.0))

    # ------------------------------------------------------------------ #
    #  Norm check                                                         #
    # ------------------------------------------------------------------ #

    def _norm_check(
        self,
        delta_flat: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """
        Check if the update norm exceeds the clip threshold.

        Returns
        -------
        is_suspicious : bool
        clipped_flat : np.ndarray
            The (possibly clipped) flat delta.
        """
        norm = float(np.linalg.norm(delta_flat))
        if norm <= self.clipping.clip_threshold:
            return False, delta_flat

        logger.warning(
            "Update norm %.4f exceeds threshold %.4f",
            norm, self.clipping.clip_threshold,
        )
        if self.clipping.clip_value is not None:
            # Clip to allowed magnitude instead of outright rejection
            scale = self.clipping.clip_value / (norm + 1e-12)
            return True, delta_flat * scale
        # No clip_value ⇒ hard rejection (caller sets weight = 0)
        return True, delta_flat

    # ------------------------------------------------------------------ #
    #  Main validation pipeline                                           #
    # ------------------------------------------------------------------ #

    def validate_updates(
        self,
        client_updates: Dict[str, List[np.ndarray]],
        data_volumes: Dict[str, int],
        server_val_data: tf.data.Dataset,
    ) -> List[ClientUpdateRecord]:
        """
        Validate every client update and assign contribution weights.

        Parameters
        ----------
        client_updates : dict[str, list[np.ndarray]]
            Mapping ``client_id → updated model weights`` (full weights,
            **not** deltas — deltas are computed internally).
        data_volumes : dict[str, int]
            Mapping ``client_id → number of local training samples``.
        server_val_data : tf.data.Dataset
            The server-side held-out validation set.

        Returns
        -------
        records : list[ClientUpdateRecord]
            One record per client, with ``contribution_weight`` set.
        """
        global_weights = self.global_model.get_weights()
        shapes = [w.shape for w in global_weights]
        global_flat = flatten_weights(global_weights)

        # ---- 0. Baseline score on server val set ---------------------- #
        baseline_score = self._evaluate(global_weights, server_val_data)
        logger.info("Baseline server score (%s): %.4f", self.eval_metric, baseline_score)

        records: List[ClientUpdateRecord] = []
        gains: List[float] = []       # for min-max normalisation later
        sims: List[float] = []
        raw_data_vols: List[int] = []
        reps: List[float] = []

        # ---- Per-client loop ----------------------------------------- #
        for cid, updated_weights in client_updates.items():
            delta = compute_update_delta(global_weights, updated_weights)
            delta_flat = flatten_weights(delta)
            norm = float(np.linalg.norm(delta_flat))

            rec = ClientUpdateRecord(client_id=cid, delta=delta, norm=norm)

            # 1.  Norm check
            is_suspicious, clipped_flat = self._norm_check(delta_flat)
            rec.is_suspicious = is_suspicious

            if is_suspicious and self.clipping.clip_value is None:
                # Hard rejection — set weight = 0, skip evaluation
                rec.rejected = True
                rec.contribution_weight = 0.0
                records.append(rec)
                gains.append(0.0)
                sims.append(0.0)
                raw_data_vols.append(data_volumes.get(cid, 0))
                reps.append(self.ledger.get(cid))
                logger.info(
                    "Client %s REJECTED (norm %.4f > %.4f, no clip_value).",
                    cid, norm, self.clipping.clip_threshold,
                )
                continue

            # Possibly overwrite delta with clipped version
            if is_suspicious:
                delta = unflatten_weights(clipped_flat, shapes)
                rec.delta = delta

            # 2.  Server-side validation gain
            temp_weights = apply_update(global_weights, delta, scale=1.0)
            new_score = self._evaluate(temp_weights, server_val_data)
            G_i = new_score - baseline_score
            rec.validation_gain = G_i

            # 3.  Similarity check
            hist_dir = self.update_history.mean_direction
            if hist_dir is not None:
                sim_i = cosine_similarity(flatten_weights(delta), hist_dir)
            else:
                sim_i = 0.5   # neutral when no history yet
            rec.similarity = sim_i

            gains.append(G_i)
            sims.append(sim_i)
            raw_data_vols.append(data_volumes.get(cid, 0))
            reps.append(self.ledger.get(cid))
            records.append(rec)

        # ---- 4. Combine into normalised contribution weights ---------- #
        n = len(records)
        if n == 0:
            return records

        arr_G = np.array(gains, dtype=np.float64)
        arr_sim = np.array(sims, dtype=np.float64)
        arr_D = _normalise_scalar_to_01(_log_scale(np.array(raw_data_vols, dtype=np.float64)))
        arr_R = np.array(reps, dtype=np.float64)          # already [0, 1]

        w = self.weights
        raw_scores = (
            w.alpha * arr_G
            + w.beta  * arr_sim
            + w.gamma * arr_D
            + w.delta * arr_R
        )

        # Normalise raw scores into [0, 1]
        c = _normalise_scalar_to_01(raw_scores)

        # Reject strongly harmful updates  (G_i < −ε)
        for idx, rec in enumerate(records):
            if rec.rejected:
                c[idx] = 0.0
                continue
            rec.raw_contribution = float(raw_scores[idx])
            rec.contribution_weight = float(c[idx])

            if rec.validation_gain < -self.harmful_threshold:
                rec.contribution_weight = 0.0
                rec.rejected = True
                logger.info(
                    "Client %s rejected — G_i=%.4f < −ε (%.4f).",
                    rec.client_id, rec.validation_gain, self.harmful_threshold,
                )

        return records

    # ------------------------------------------------------------------ #
    #  5. Weighted aggregation                                            #
    # ------------------------------------------------------------------ #

    def aggregate_weighted(
        self,
        records: List[ClientUpdateRecord],
        global_weights: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        Contribution-weighted aggregation of client deltas.

        ``new_global = global + Σ_i  (c_i / Σ c_j) · delta_i``

        Parameters
        ----------
        records : list[ClientUpdateRecord]
            Output of ``validate_updates``.
        global_weights : list[np.ndarray] | None
            If ``None``, reads from ``self.global_model``.

        Returns
        -------
        new_global_weights : list[np.ndarray]
        """
        if global_weights is None:
            global_weights = self.global_model.get_weights()

        # Filter to accepted updates with positive weight
        active = [(r.delta, r.contribution_weight) for r in records
                  if not r.rejected and r.contribution_weight > 0]

        if not active:
            logger.warning("No valid updates this round — global model unchanged.")
            return global_weights

        total_c = sum(c for _, c in active)
        aggregated_delta = [np.zeros_like(w) for w in global_weights]

        for delta, c_i in active:
            weight = c_i / total_c
            for idx, d in enumerate(delta):
                aggregated_delta[idx] += weight * d

        new_weights = apply_update(global_weights, aggregated_delta)

        # Push this aggregated delta into the history for future similarity
        self.update_history.push(flatten_weights(aggregated_delta))

        return new_weights

    # ------------------------------------------------------------------ #
    #  6. Reputation feedback                                             #
    # ------------------------------------------------------------------ #

    def update_reputations(
        self,
        records: List[ClientUpdateRecord],
    ) -> None:
        """
        Feed observed validation gains and contribution weights back into
        the reputation ledger.

        - Clients with ``G_i > 0`` and ``c_i > 0`` are rewarded.
        - Clients with ``G_i < 0`` or no contribution are penalised.
        - Rejected clients receive a stronger penalty.
        """
        for rec in records:
            if rec.rejected:
                # Stronger penalty for rejected / harmful updates
                self.ledger.update(rec.client_id, update_was_beneficial=False)
                self.ledger.update(rec.client_id, update_was_beneficial=False)
                logger.debug("Reputation double-penalty for %s (rejected).", rec.client_id)
            elif rec.validation_gain > 0 and rec.contribution_weight > 0:
                self.ledger.update(rec.client_id, update_was_beneficial=True)
            else:
                self.ledger.update(rec.client_id, update_was_beneficial=False)


# ====================================================================== #
#  5.  FULL ROUND RUNNER  (wraps Part 1 selection + Part 2 validation)    #
# ====================================================================== #

class ValidatedFederatedRound:
    """
    End-to-end runner for one federated round that integrates:

    * **Part 1** — Enhanced client selection
    * **Part 2** — Update validation & contribution-weighted aggregation

    Parameters
    ----------
    global_model : tf.keras.Model
        The current global model (mutated in-place each round).
    clients : list[FederatedClient]
        Full client pool.
    selector
        An ``EnhancedClientSelector`` instance (Part 1).
    validator : UpdateValidator
        The update-validation & scoring engine (Part 2).
    local_epochs : int
        Local training epochs per client.
    local_batch_size : int
        Batch size for local training.
    learning_rate : float
        Client-side optimiser learning rate.
    """

    def __init__(
        self,
        global_model: tf.keras.Model,
        clients: List[FederatedClient],
        selector,                          # EnhancedClientSelector
        validator: UpdateValidator,
        local_epochs: int = 1,
        local_batch_size: int = 32,
        learning_rate: float = 1e-4,
    ) -> None:
        self.global_model = global_model
        self.clients = {c.client_id: c for c in clients}
        self.selector = selector
        self.validator = validator
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.learning_rate = learning_rate

    # ------------------------------------------------------------------ #
    #  Local training helper                                              #
    # ------------------------------------------------------------------ #

    def _local_train(
        self,
        client: FederatedClient,
        global_weights: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], int]:
        """
        Run local training on *client* starting from *global_weights*.

        Returns ``(updated_weights, num_samples)``.
        """
        local_model = tf.keras.models.clone_model(self.global_model)
        local_model.build(self.global_model.input_shape)
        local_model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        local_model.set_weights(global_weights)

        if client.local_data is None:
            logger.warning("Client %s has no data — returning global weights.", client.client_id)
            return global_weights, 0

        dataset = client.local_data.batch(self.local_batch_size)
        local_model.fit(dataset, epochs=self.local_epochs, verbose=0)
        return local_model.get_weights(), client.metrics.data_volume

    # ------------------------------------------------------------------ #
    #  Single round                                                       #
    # ------------------------------------------------------------------ #

    def execute_round(
        self,
        current_round: int,
        server_val_data: tf.data.Dataset,
    ) -> Dict:
        """
        Execute one complete federated round.

        1. Select clients  (Part 1).
        2. Distribute global weights & run local training.
        3. Validate updates & compute contribution weights (Part 2).
        4. Weighted aggregation.
        5. Update reputation ledger.

        Returns
        -------
        round_info : dict
            Summary including selected clients, records, and new accuracy.
        """
        # 1. Client selection
        selected = self.selector.select(current_round=current_round)
        global_weights = self.global_model.get_weights()

        # 2. Local training
        client_updates: Dict[str, List[np.ndarray]] = {}
        data_volumes: Dict[str, int] = {}
        for client in selected:
            updated_w, n_samples = self._local_train(client, global_weights)
            client_updates[client.client_id] = updated_w
            data_volumes[client.client_id] = max(n_samples, 1)
            client.metrics.last_selected_round = current_round

        # 3. Validate updates & contribution scoring
        records = self.validator.validate_updates(
            client_updates=client_updates,
            data_volumes=data_volumes,
            server_val_data=server_val_data,
        )

        # 4. Weighted aggregation
        new_weights = self.validator.aggregate_weighted(records, global_weights)
        self.global_model.set_weights(new_weights)

        # Also update the validator's reference model
        self.validator.global_model.set_weights(new_weights)

        # 5. Reputation feedback
        self.validator.update_reputations(records)

        # Evaluate new global model
        result = self.global_model.evaluate(
            server_val_data.batch(self.local_batch_size),
            verbose=0,
            return_dict=True,
        )
        acc = result.get("accuracy", 0.0)
        logger.info("Round %d — post-aggregation accuracy: %.4f", current_round, acc)

        return {
            "round": current_round,
            "selected": [c.client_id for c in selected],
            "records": records,
            "global_accuracy": acc,
        }

    # ------------------------------------------------------------------ #
    #  Multi-round loop                                                   #
    # ------------------------------------------------------------------ #

    def run(
        self,
        num_rounds: int,
        server_val_data: tf.data.Dataset,
    ) -> Dict[str, list]:
        """
        Run *num_rounds* federated rounds end-to-end.

        Returns a history dict similar to ``FederatedRoundRunner.run()``.
        """
        history: Dict[str, list] = {
            "round": [],
            "global_accuracy": [],
            "selected_clients": [],
            "contribution_weights": [],
        }

        for t in range(1, num_rounds + 1):
            logger.info("=" * 60)
            logger.info("ROUND %d / %d", t, num_rounds)
            logger.info("=" * 60)

            info = self.execute_round(t, server_val_data)
            history["round"].append(t)
            history["global_accuracy"].append(info["global_accuracy"])
            history["selected_clients"].append(info["selected"])
            history["contribution_weights"].append(
                {r.client_id: r.contribution_weight for r in info["records"]}
            )

        logger.info("Federated training complete — %d rounds.", num_rounds)
        return history


# ====================================================================== #
#  DEMO / SMOKE-TEST  (no model or data — synthetic only)                 #
# ====================================================================== #

if __name__ == "__main__":
    print("\n===  Update Validation & Contribution Weighing — Demo  ===\n")

    # ---- synthetic setup --------------------------------------------- #
    np.random.seed(42)

    NUM_CLIENTS = 8
    NUM_LAYERS = 3                 # pretend model has 3 weight tensors
    LAYER_SHAPE = (4, 4)          # small for demo

    # Fake "global weights"
    global_weights = [np.random.randn(*LAYER_SHAPE).astype(np.float32)
                      for _ in range(NUM_LAYERS)]

    # Build client updates (simulate deltas of varying quality)
    client_updates: Dict[str, List[np.ndarray]] = {}
    data_volumes: Dict[str, int] = {}

    for i in range(NUM_CLIENTS):
        cid = f"client_{i:02d}"
        # Varying quality: some improve, some degrade, one is huge (suspicious)
        if i == 5:
            # suspiciously large update
            noise_scale = 50.0
        elif i == 7:
            # harmful update (large negative impact simulated later)
            noise_scale = 2.0
        else:
            noise_scale = 0.3

        delta = [np.random.randn(*LAYER_SHAPE).astype(np.float32) * noise_scale
                 for _ in range(NUM_LAYERS)]
        client_updates[cid] = [g + d for g, d in zip(global_weights, delta)]
        data_volumes[cid] = int(np.random.randint(200, 5000))

    # ---- Reputation ledger ------------------------------------------- #
    ledger = ReputationLedger()
    for cid in client_updates:
        ledger.register(cid)

    # ---- Build a minimal "model" for the demo (no real inference) ---- #
    # We can't call _evaluate without a real Keras model, so we'll
    # exercise everything *except* the TF evaluation path by subclassing.

    class _DemoValidator(UpdateValidator):
        """Override _evaluate to return a synthetic score for the demo."""

        def _evaluate(self, model_weights, val_data):
            # Score = negative of mean absolute weight magnitude
            # (smaller weights ⇒ higher "score", just for illustration)
            flat = flatten_weights(model_weights)
            return float(1.0 / (1.0 + np.mean(np.abs(flat))))

    validator = _DemoValidator(
        global_model=None,      # not used in our overridden _evaluate
        reputation_ledger=ledger,
        weights=ContributionWeights(alpha=0.40, beta=0.15, gamma=0.20, delta=0.25),
        clipping=ClippingConfig(clip_threshold=10.0, clip_value=5.0),
        harmful_threshold=0.02,
    )

    # Monkey-patch get_weights so aggregate_weighted works without a real model
    class _FakeModel:
        def get_weights(self):
            return global_weights
    validator.global_model = _FakeModel()

    # ---- Create a fake tf.data.Dataset (unused by our mock) ---------- #
    fake_val = tf.data.Dataset.from_tensor_slices(
        (np.zeros((2, 4)), np.zeros((2,)))
    )

    # ---- Run validation pipeline ------------------------------------- #
    records = validator.validate_updates(
        client_updates=client_updates,
        data_volumes=data_volumes,
        server_val_data=fake_val,
    )

    print(f"{'Client':<14} {'Norm':>8} {'G_i':>8} {'Sim':>8} "
          f"{'c_i':>8} {'Status':<12}")
    print("-" * 62)
    for rec in records:
        status = "REJECTED" if rec.rejected else (
            "SUSPICIOUS" if rec.is_suspicious else "ok"
        )
        print(f"{rec.client_id:<14} {rec.norm:>8.3f} {rec.validation_gain:>8.4f} "
              f"{rec.similarity:>8.4f} {rec.contribution_weight:>8.4f} {status:<12}")

    # ---- Weighted aggregation ---------------------------------------- #
    new_weights = validator.aggregate_weighted(records, global_weights)
    print(f"\nAggregated delta norm: "
          f"{np.linalg.norm(flatten_weights(compute_update_delta(global_weights, new_weights))):.4f}")

    # ---- Reputation update ------------------------------------------- #
    validator.update_reputations(records)
    print("\n--- Reputation Ledger After Round ---")
    for cid, rep in sorted(ledger.summary().items()):
        print(f"  {cid}: {rep:.4f}")
