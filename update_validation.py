

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



        Reduced from 10 → 3 to save ~250 MB (each vector is ~36 MB



        for EfficientNetB2).



    """







    def __init__(self, max_history: int = 1) -> None:



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



        self._eval_model: Optional[tf.keras.Model] = None



        self._val_data_batched = None







    # ------------------------------------------------------------------ #



    #  Evaluation helper                                                  #



    # ------------------------------------------------------------------ #







    def _evaluate(



        self,



        model_weights: List[np.ndarray],



        val_data: tf.data.Dataset,



    ) -> float:



        """



        Evaluate *model_weights* on *val_data* using a cached temporary



        model (avoids clone+build+compile overhead on every call).



        """



        if self._eval_model is None:



            self._eval_model = tf.keras.models.clone_model(self.global_model)



            self._eval_model.build(self.global_model.input_shape)



            self._eval_model.compile(



                optimizer="adam",



                loss="binary_crossentropy",



                metrics=["accuracy"],



            )



        if self._val_data_batched is None:



            self._val_data_batched = (



                val_data.batch(self.batch_size).prefetch(1)



            )



        self._eval_model.set_weights(model_weights)



        results = self._eval_model.evaluate(



            self._val_data_batched, verbose=0, return_dict=True,



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



        """



        norm = float(np.linalg.norm(delta_flat))



        if norm <= self.clipping.clip_threshold:



            return False, delta_flat







        logger.warning(



            "Update norm %.4f exceeds threshold %.4f",



            norm, self.clipping.clip_threshold,



        )



        if self.clipping.clip_value is not None:



            scale = self.clipping.clip_value / (norm + 1e-12)



            return True, delta_flat * scale



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



        """



        global_weights = self.global_model.get_weights()



        shapes = [w.shape for w in global_weights]



        global_flat = flatten_weights(global_weights)







        # ---- 0. Baseline score on server val set ---------------------- #



        baseline_score = self._evaluate(global_weights, server_val_data)



        logger.info("Baseline server score (%s): %.4f", self.eval_metric, baseline_score)







        records: List[ClientUpdateRecord] = []



        gains: List[float] = []



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



                rec.rejected = True



                rec.contribution_weight = 0.0



                records.append(rec)



                gains.append(0.0)



                sims.append(0.0)



                raw_data_vols.append(data_volumes.get(cid, 0))



                reps.append(self.ledger.get(cid))



                logger.debug(



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



                sim_i = 0.5



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



        arr_R = np.array(reps, dtype=np.float64)







        w = self.weights



        raw_scores = (



            w.alpha * arr_G



            + w.beta  * arr_sim



            + w.gamma * arr_D



            + w.delta * arr_R



        )







        # Normalise raw scores via softmax (ensures all non-rejected



        # clients get non-zero weights even when scores are similar;



        # min-max returns all-zeros when scores are identical, causing



        # aggregate_weighted to silently no-op).



        _shifted = raw_scores - np.max(raw_scores)  # numerical stability



        _exp = np.exp(_shifted)



        c = _exp / (_exp.sum() + 1e-12)







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



                logger.debug(



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



        """



        if global_weights is None:



            global_weights = self.global_model.get_weights()







        active = [(r.delta, r.contribution_weight) for r in records



                  if not r.rejected and r.contribution_weight > 0]







        if not active:



            logger.warning("No valid updates this round — global model unchanged.")



            return global_weights







        total_c = sum(c for _, c in active)



        aggregated_delta = [np.zeros(w.shape, dtype=np.float64) for w in global_weights]







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



        """



        for rec in records:



            if rec.rejected:



                self.ledger.update(rec.client_id, update_was_beneficial=False)



                self.ledger.update(rec.client_id, update_was_beneficial=False)



                logger.debug("Reputation double-penalty for %s (rejected).", rec.client_id)



            elif rec.validation_gain > 0 and rec.contribution_weight > 0:



                self.ledger.update(rec.client_id, update_was_beneficial=True)



            else:



                self.ledger.update(rec.client_id, update_was_beneficial=False)
