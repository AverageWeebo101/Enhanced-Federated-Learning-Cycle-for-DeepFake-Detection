"""
Server-Side Knowledge Distillation
===================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

After federated aggregation (Parts 1 & 2), the server refines the global
model by distilling knowledge from the contribution-weighted ensemble of
client models into the global student model, using unlabelled proxy data
from FF++ c23.

Pipeline
--------
1.  **Build teacher logits** — weighted average of per-client logits on
    every proxy input, where the weights are the contribution scores
    ``{c_i}`` from Part 2.
2.  **Distillation loop** — minimise ``λ · T² · KL(p_teach ‖ p_stud)``
    (soft targets) plus an optional ``(1−λ) · CE`` supervised term when
    labelled data is available.
3.  Return the distilled global model.

Imports shared types from ``enhanced_client_selection.py`` and helpers
from ``update_validation.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# ---------- shared types from Parts 1 & 2 ----------------------------- #
from enhanced_client_selection import FederatedClient, ReputationLedger

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
class DistillationConfig:
    """
    Hyper-parameters for server-side knowledge distillation.

    Parameters
    ----------
    temperature : float
        Softmax temperature ``T`` — higher values produce softer
        probability distributions, transferring more "dark knowledge".
    lam : float
        Interpolation weight ``λ`` between the distillation loss and the
        optional supervised cross-entropy loss:
        ``L_total = λ · L_KD  +  (1 − λ) · L_sup``
    epochs : int
        Number of distillation training epochs.
    batch_size : int
        Batch size for iterating over proxy / supervised data.
    learning_rate : float
        Learning rate for the distillation optimiser.
    """
    temperature: float = 3.0
    lam: float = 0.7
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4


# ====================================================================== #
#  2.  TEACHER LOGIT BUILDER                                              #
# ====================================================================== #

class TeacherEnsemble:
    """
    Builds a *virtual* teacher by computing the contribution-weighted
    average of per-client logits for every proxy input.

    The teacher is never materialised as a single model; instead its
    output logits are pre-computed and cached so the distillation loop
    can iterate over them efficiently.

    Parameters
    ----------
    global_model : tf.keras.Model
        Used as the architecture template for loading client weights.
    client_weights : dict[str, list[np.ndarray]]
        ``{client_id: model_weights}`` for every participating client.
    contribution_weights : dict[str, float]
        ``{client_id: c_i}`` from Part 2 — only clients with ``c_i > 0``
        are included.
    """

    def __init__(
        self,
        global_model: tf.keras.Model,
        client_weights: Dict[str, List[np.ndarray]],
        contribution_weights: Dict[str, float],
    ) -> None:
        self.global_model = global_model
        # Filter to clients that actually contribute
        self.client_weights = {
            cid: w for cid, w in client_weights.items()
            if contribution_weights.get(cid, 0.0) > 0
        }
        self.contribution_weights = {
            cid: contribution_weights[cid]
            for cid in self.client_weights
        }
        total_c = sum(self.contribution_weights.values())
        # Normalise so weights sum to 1
        self._norm_weights = {
            cid: c / total_c for cid, c in self.contribution_weights.items()
        }
        logger.info(
            "TeacherEnsemble: %d client(s), normalised weights: %s",
            len(self._norm_weights),
            {k: round(v, 4) for k, v in self._norm_weights.items()},
        )

    # ------------------------------------------------------------------ #
    #  Logit-model builder (creates a model that outputs pre-softmax)     #
    # ------------------------------------------------------------------ #

    def _build_logit_model(
        self,
        weights: List[np.ndarray],
    ) -> tf.keras.Model:
        """
        Clone the global model, load *weights*, and strip the final
        activation so the output is raw **logits**.

        Strategy: rebuild the architecture layer-by-layer. For the last
        Dense layer, replace its activation with ``linear``.  This avoids
        fragile graph surgery and works across Sequential / Functional
        models in all TF 2.x versions.
        """
        # --- Rebuild architecture with linear final activation --------- #
        logit_model = self._rebuild_with_linear_output(self.global_model)
        logit_model.set_weights(weights)
        return logit_model

    @staticmethod
    def _rebuild_with_linear_output(
        ref_model: tf.keras.Model,
    ) -> tf.keras.Model:
        """
        Rebuild *ref_model* identically, except the last ``Dense`` layer
        uses ``activation='linear'`` so the output is raw logits.

        Falls back to the original architecture when no Dense layer is
        found (e.g. the model already produces logits).
        """
        # Walk layers to find the last Dense
        last_dense_idx: Optional[int] = None
        for idx, layer in enumerate(ref_model.layers):
            if isinstance(layer, tf.keras.layers.Dense):
                last_dense_idx = idx

        # Identify which layers are *not* InputLayer
        non_input_layers = [
            (idx, layer) for idx, layer in enumerate(ref_model.layers)
            if not isinstance(layer, tf.keras.layers.InputLayer)
        ]

        # Build input
        input_shape = ref_model.input_shape
        if isinstance(input_shape, tuple):
            # Strip the batch dimension
            input_shape = input_shape[1:]
        x = inp = tf.keras.layers.Input(shape=input_shape)

        for idx, layer in non_input_layers:
            if idx == last_dense_idx:
                # Replace activation with linear
                cfg = layer.get_config()
                cfg["name"] = cfg["name"] + "_logits"
                cfg["activation"] = "linear"
                new_layer = tf.keras.layers.Dense.from_config(cfg)
                x = new_layer(x)
            else:
                # Re-use the same layer class & config
                cloned = layer.__class__.from_config(layer.get_config())
                x = cloned(x)

        logit_model = tf.keras.Model(inputs=inp, outputs=x)

        # Transfer weights layer by layer (names may differ for the
        # modified Dense, so match by position among non-input layers).
        ref_non_input = [l for l in ref_model.layers
                         if not isinstance(l, tf.keras.layers.InputLayer)]
        logit_non_input = [l for l in logit_model.layers
                           if not isinstance(l, tf.keras.layers.InputLayer)]
        for src, dst in zip(ref_non_input, logit_non_input):
            ws = src.get_weights()
            if ws:
                dst.set_weights(ws)

        return logit_model

    # ------------------------------------------------------------------ #
    #  Compute teacher logits for a batch                                 #
    # ------------------------------------------------------------------ #

    def compute_teacher_logits_batch(
        self,
        x_batch: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the contribution-weighted average teacher logits for
        *x_batch*.

        ``z_teach(x) = Σ_i  w_i · logits(M_i, x)``
        """
        weighted_logits = None
        for cid, weights in self.client_weights.items():
            logit_model = self._build_logit_model(weights)
            logits = logit_model(x_batch, training=False)     # (B, num_classes) or (B, 1)
            scaled = tf.cast(logits, tf.float32) * self._norm_weights[cid]
            if weighted_logits is None:
                weighted_logits = scaled
            else:
                weighted_logits = weighted_logits + scaled
        return weighted_logits                                 # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Pre-compute teacher logits for entire proxy dataset                #
    # ------------------------------------------------------------------ #

    def precompute_teacher_logits(
        self,
        proxy_data: tf.data.Dataset,
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-compute teacher logits for every sample in *proxy_data*.

        Parameters
        ----------
        proxy_data : tf.data.Dataset
            Must yield ``(images,)`` or ``(images, _)`` — labels are
            ignored.
        batch_size : int

        Returns
        -------
        all_inputs : np.ndarray   — shape ``(N, ...)``
        all_logits : np.ndarray   — shape ``(N, num_classes)`` or ``(N, 1)``
        """
        all_inputs: List[np.ndarray] = []
        all_logits: List[np.ndarray] = []

        batched = proxy_data.batch(batch_size)
        for batch in batched:
            if isinstance(batch, (list, tuple)):
                x_batch = batch[0]
            else:
                x_batch = batch

            teacher_logits = self.compute_teacher_logits_batch(x_batch)
            all_inputs.append(x_batch.numpy())
            all_logits.append(teacher_logits.numpy())

        return np.concatenate(all_inputs), np.concatenate(all_logits)


# ====================================================================== #
#  3.  DISTILLATION LOSSES                                                #
# ====================================================================== #

def distillation_loss(
    teacher_logits: tf.Tensor,
    student_logits: tf.Tensor,
    temperature: float,
) -> tf.Tensor:
    """
    Compute the knowledge-distillation loss:

        ``L_KD = T² · KL( softmax(z_teach / T)  ‖  softmax(z_stud / T) )``

    Works for both multi-class (last dim > 1) and binary (last dim = 1).
    """
    # Determine if binary or multi-class
    num_outputs = teacher_logits.shape[-1]

    if num_outputs == 1:
        # Binary case: use sigmoid + binary KL
        p_teach = tf.sigmoid(teacher_logits / temperature)
        p_stud  = tf.sigmoid(student_logits / temperature)
        # Binary KL divergence  KL(p || q) = p·log(p/q) + (1-p)·log((1-p)/(1-q))
        eps = 1e-7
        p_teach = tf.clip_by_value(p_teach, eps, 1.0 - eps)
        p_stud  = tf.clip_by_value(p_stud,  eps, 1.0 - eps)
        kl = (
            p_teach * tf.math.log(p_teach / p_stud)
            + (1.0 - p_teach) * tf.math.log((1.0 - p_teach) / (1.0 - p_stud))
        )
        return temperature ** 2 * tf.reduce_mean(kl)
    else:
        # Multi-class case: standard KL on softmax outputs
        p_teach = tf.nn.softmax(teacher_logits / temperature)
        log_p_stud = tf.nn.log_softmax(student_logits / temperature)
        # KL(p_teach || p_stud) = Σ p_teach · (log p_teach − log p_stud)
        kl = tf.reduce_sum(
            p_teach * (tf.math.log(p_teach + 1e-12) - log_p_stud),
            axis=-1,
        )
        return temperature ** 2 * tf.reduce_mean(kl)


def supervised_loss(
    student_logits: tf.Tensor,
    labels: tf.Tensor,
) -> tf.Tensor:
    """
    Standard supervised cross-entropy loss.

    Handles binary (1-unit output, BCE) and multi-class (softmax CE)
    automatically.
    """
    num_outputs = student_logits.shape[-1]
    if num_outputs == 1:
        # Ensure labels have the same rank as logits: (B,) → (B, 1)
        labels = tf.cast(labels, tf.float32)
        if len(labels.shape) < len(student_logits.shape):
            labels = tf.expand_dims(labels, -1)
        return tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                labels, student_logits, from_logits=True,
            )
        )
    else:
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, student_logits, from_logits=True,
            )
        )


# ====================================================================== #
#  4.  KNOWLEDGE DISTILLATION ENGINE                                      #
# ====================================================================== #

class KnowledgeDistiller:
    """
    Performs server-side knowledge distillation from a weighted ensemble
    of client models (teacher) into the aggregated global model (student).

    Parameters
    ----------
    global_model : tf.keras.Model
        The global/student model — **modified in-place**.
    teacher : TeacherEnsemble
        The virtual teacher (built from client logits).
    config : DistillationConfig
        Temperature, λ, epochs, batch size, learning rate.
    """

    def __init__(
        self,
        global_model: tf.keras.Model,
        teacher: TeacherEnsemble,
        config: Optional[DistillationConfig] = None,
    ) -> None:
        self.global_model = global_model
        self.teacher = teacher
        self.config = config or DistillationConfig()
        self.optimizer = tf.keras.optimizers.Adam(self.config.learning_rate)

    # ------------------------------------------------------------------ #
    #  Build a "logit model" view of the global student                   #
    # ------------------------------------------------------------------ #

    def _build_student_logit_model(self) -> tf.keras.Model:
        """
        Return a version of the global model that outputs raw logits.

        Uses the same stripping logic as ``TeacherEnsemble``.
        """
        return self.teacher._build_logit_model(self.global_model.get_weights())

    # ------------------------------------------------------------------ #
    #  Single training step                                               #
    # ------------------------------------------------------------------ #

    @tf.function
    def _train_step_kd_only(
        self,
        x_batch: tf.Tensor,
        teacher_logits: tf.Tensor,
        student_model: tf.keras.Model,
        temperature: float,
    ) -> tf.Tensor:
        """Pure distillation step (no supervised term)."""
        with tf.GradientTape() as tape:
            student_logits = student_model(x_batch, training=True)
            loss = distillation_loss(teacher_logits, student_logits, temperature)
        grads = tape.gradient(loss, student_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, student_model.trainable_variables)
        )
        return loss

    @tf.function
    def _train_step_combined(
        self,
        x_proxy: tf.Tensor,
        teacher_logits: tf.Tensor,
        x_sup: tf.Tensor,
        y_sup: tf.Tensor,
        student_model: tf.keras.Model,
        temperature: float,
        lam: float,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Combined distillation + supervised step."""
        with tf.GradientTape() as tape:
            # KD part
            stud_logits_proxy = student_model(x_proxy, training=True)
            l_kd = distillation_loss(teacher_logits, stud_logits_proxy, temperature)

            # Supervised part
            stud_logits_sup = student_model(x_sup, training=True)
            l_sup = supervised_loss(stud_logits_sup, y_sup)

            l_total = lam * l_kd + (1.0 - lam) * l_sup

        grads = tape.gradient(l_total, student_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, student_model.trainable_variables)
        )
        return l_total, l_kd, l_sup

    # ------------------------------------------------------------------ #
    #  Full distillation loop                                             #
    # ------------------------------------------------------------------ #

    def distill(
        self,
        proxy_data: tf.data.Dataset,
        supervised_data: Optional[tf.data.Dataset] = None,
    ) -> Dict[str, List[float]]:
        """
        Run the full distillation loop.

        Parameters
        ----------
        proxy_data : tf.data.Dataset
            Unlabelled proxy inputs from FF++ c23.
            Yields ``(images,)`` or ``(images, _)`` — labels ignored.
        supervised_data : tf.data.Dataset | None
            Optional labelled data yielding ``(images, labels)``.
            When provided, the combined loss
            ``λ · L_KD + (1−λ) · L_sup`` is used.

        Returns
        -------
        history : dict
            ``{"epoch": [...], "loss_total": [...], "loss_kd": [...],
               "loss_sup": [...]}``
        """
        cfg = self.config
        T = cfg.temperature
        lam = cfg.lam

        history: Dict[str, List[float]] = {
            "epoch": [],
            "loss_total": [],
            "loss_kd": [],
            "loss_sup": [],
        }

        # --- Pre-compute teacher logits ------------------------------- #
        logger.info("Pre-computing teacher logits (T=%.1f) …", T)
        proxy_inputs, teacher_logits_all = self.teacher.precompute_teacher_logits(
            proxy_data, batch_size=cfg.batch_size,
        )
        logger.info(
            "Teacher logits ready — %d samples, shape %s",
            len(proxy_inputs), teacher_logits_all.shape,
        )

        # Wrap into a tf.data.Dataset for batching
        proxy_ds = (
            tf.data.Dataset.from_tensor_slices((proxy_inputs, teacher_logits_all))
            .shuffle(buffer_size=len(proxy_inputs))
            .batch(cfg.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        # Prepare supervised data iterator (if available)
        if supervised_data is not None:
            sup_ds = (
                supervised_data
                .shuffle(buffer_size=1000)
                .batch(cfg.batch_size)
                .repeat()                       # cycle forever — we zip with proxy
                .prefetch(tf.data.AUTOTUNE)
            )
            sup_iter = iter(sup_ds)
        else:
            sup_iter = None

        # --- Build student logit model -------------------------------- #
        # We train a clone that shares the same architecture & initial
        # weights, then copy weights back to self.global_model at the end.
        student = self._build_student_logit_model()

        # --- Distillation epochs -------------------------------------- #
        for epoch in range(1, cfg.epochs + 1):
            epoch_loss_total = []
            epoch_loss_kd = []
            epoch_loss_sup = []

            for batch in proxy_ds:
                x_proxy_batch, teach_logits_batch = batch

                if sup_iter is not None:
                    # Combined loss
                    try:
                        x_sup_batch, y_sup_batch = next(sup_iter)
                    except StopIteration:
                        sup_iter = iter(sup_ds)
                        x_sup_batch, y_sup_batch = next(sup_iter)

                    l_total, l_kd, l_sup = self._train_step_combined(
                        x_proxy_batch, teach_logits_batch,
                        x_sup_batch, y_sup_batch,
                        student, T, lam,
                    )
                    epoch_loss_total.append(float(l_total))
                    epoch_loss_kd.append(float(l_kd))
                    epoch_loss_sup.append(float(l_sup))
                else:
                    # KD only
                    l_kd = self._train_step_kd_only(
                        x_proxy_batch, teach_logits_batch,
                        student, T,
                    )
                    epoch_loss_total.append(float(l_kd))
                    epoch_loss_kd.append(float(l_kd))
                    epoch_loss_sup.append(0.0)

            mean_total = float(np.mean(epoch_loss_total))
            mean_kd    = float(np.mean(epoch_loss_kd))
            mean_sup   = float(np.mean(epoch_loss_sup))

            history["epoch"].append(epoch)
            history["loss_total"].append(mean_total)
            history["loss_kd"].append(mean_kd)
            history["loss_sup"].append(mean_sup)

            logger.info(
                "Distillation epoch %d/%d — L_total=%.5f  L_KD=%.5f  L_sup=%.5f",
                epoch, cfg.epochs, mean_total, mean_kd, mean_sup,
            )

        # --- Copy distilled weights back to global model -------------- #
        self.global_model.set_weights(student.get_weights())
        logger.info("Distilled weights applied to global model.")

        return history


# ====================================================================== #
#  5.  CONVENIENCE: one-call distillation after a federated round         #
# ====================================================================== #

def run_distillation_round(
    global_model: tf.keras.Model,
    client_weights: Dict[str, List[np.ndarray]],
    contribution_weights: Dict[str, float],
    proxy_data: tf.data.Dataset,
    supervised_data: Optional[tf.data.Dataset] = None,
    config: Optional[DistillationConfig] = None,
) -> Dict[str, List[float]]:
    """
    One-liner helper that creates the teacher ensemble, distiller,
    and runs the distillation loop.

    Parameters
    ----------
    global_model : tf.keras.Model
        Aggregated global model (modified in-place).
    client_weights : dict[str, list[np.ndarray]]
        Per-client model weights.
    contribution_weights : dict[str, float]
        Per-client contribution scores ``c_i`` from Part 2.
    proxy_data : tf.data.Dataset
        Unlabelled proxy inputs (FF++ c23).
    supervised_data : tf.data.Dataset | None
        Optional labelled data for the combined loss.
    config : DistillationConfig | None
        Hyper-parameters (defaults are sensible).

    Returns
    -------
    history : dict
    """
    config = config or DistillationConfig()

    teacher = TeacherEnsemble(
        global_model=global_model,
        client_weights=client_weights,
        contribution_weights=contribution_weights,
    )
    distiller = KnowledgeDistiller(
        global_model=global_model,
        teacher=teacher,
        config=config,
    )
    return distiller.distill(proxy_data, supervised_data)


# ====================================================================== #
#  DEMO / SMOKE-TEST  (synthetic data — no real model needed)             #
# ====================================================================== #

if __name__ == "__main__":
    print("\n===  Server-Side Knowledge Distillation — Demo  ===\n")

    np.random.seed(42)
    tf.random.set_seed(42)

    # ---- 1. Build a tiny model to act as global / student ------------ #
    INPUT_DIM = 16
    NUM_CLASSES = 1          # binary (DeepFake vs Real)
    NUM_PROXY = 200
    NUM_SUP = 60
    NUM_CLIENTS = 4

    global_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_DIM,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid"),
    ])
    global_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Global model params: {global_model.count_params()}")

    # ---- 2. Synthetic client models (slight perturbations) ----------- #
    base_weights = global_model.get_weights()
    client_weights: Dict[str, List[np.ndarray]] = {}
    contribution_weights: Dict[str, float] = {}

    for i in range(NUM_CLIENTS):
        cid = f"client_{i:02d}"
        noise = [np.random.randn(*w.shape).astype(np.float32) * 0.05
                 for w in base_weights]
        client_weights[cid] = [w + n for w, n in zip(base_weights, noise)]
        contribution_weights[cid] = np.random.uniform(0.3, 1.0)

    print(f"Contribution weights: {contribution_weights}")

    # ---- 3. Synthetic proxy & supervised data ------------------------ #
    proxy_x = np.random.randn(NUM_PROXY, INPUT_DIM).astype(np.float32)
    proxy_data = tf.data.Dataset.from_tensor_slices(proxy_x)

    sup_x = np.random.randn(NUM_SUP, INPUT_DIM).astype(np.float32)
    sup_y = np.random.randint(0, 2, size=(NUM_SUP,)).astype(np.float32)
    supervised_data = tf.data.Dataset.from_tensor_slices((sup_x, sup_y))

    # ---- 4. Run distillation (KD only) ------------------------------- #
    print("\n--- KD-only distillation ---")
    history_kd = run_distillation_round(
        global_model=global_model,
        client_weights=client_weights,
        contribution_weights=contribution_weights,
        proxy_data=proxy_data,
        supervised_data=None,              # no supervised term
        config=DistillationConfig(
            temperature=3.0,
            lam=0.7,
            epochs=3,
            batch_size=32,
            learning_rate=1e-3,
        ),
    )
    for ep, lt, lk in zip(history_kd["epoch"], history_kd["loss_total"],
                          history_kd["loss_kd"]):
        print(f"  Epoch {ep}: L_total={lt:.5f}  L_KD={lk:.5f}")

    # ---- 5. Run distillation (combined KD + supervised) -------------- #
    print("\n--- Combined KD + supervised distillation ---")
    # Reset model to base weights for a fair comparison
    global_model.set_weights(base_weights)

    history_comb = run_distillation_round(
        global_model=global_model,
        client_weights=client_weights,
        contribution_weights=contribution_weights,
        proxy_data=proxy_data,
        supervised_data=supervised_data,
        config=DistillationConfig(
            temperature=3.0,
            lam=0.7,
            epochs=3,
            batch_size=32,
            learning_rate=1e-3,
        ),
    )
    for ep, lt, lk, ls in zip(
        history_comb["epoch"], history_comb["loss_total"],
        history_comb["loss_kd"], history_comb["loss_sup"],
    ):
        print(f"  Epoch {ep}: L_total={lt:.5f}  L_KD={lk:.5f}  L_sup={ls:.5f}")

    print("\nDone.")
