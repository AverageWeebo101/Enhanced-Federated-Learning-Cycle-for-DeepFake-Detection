"""
Client Reputation and Client Ledger
====================================
Part of: Enhanced Federated Learning Cycle for DeepFake Detection (Thesis)

Extends the basic ``ReputationLedger`` from Part 1 with:

* **Validation-gain-based reputation update** — reputation increases
  proportionally to the measured validation gain ``G_i`` when it exceeds
  a threshold ``θ``, and holds steady (or decays) otherwise.
* **Persistent ledger** with full round-by-round history, JSON
  serialisation, and audit trail.
* **Decay & floor mechanics** — inactive or consistently harmful clients
  slowly lose reputation; a configurable floor prevents permanent
  exclusion.
* **Integration helpers** that plug directly into Parts 1–3.

Pseudo-code from the thesis proposal is faithfully reproduced.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------- shared types from Part 1 ---------------------------------- #
from enhanced_client_selection import ReputationLedger

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
class ReputationConfig:
    """
    Hyper-parameters for the gain-based reputation update rule.

    Parameters
    ----------
    theta : float
        Validation-gain threshold ``θ``.  An update is considered *valid*
        only when ``G_i > θ``.
    gamma : float
        Reputation update factor ``γ``.  Controls how strongly a positive
        gain boosts reputation.
    decay_rate : float
        Per-round multiplicative decay applied to clients that did **not**
        participate (models staleness into reputation).  Set to ``1.0`` to
        disable decay.
    floor : float
        Minimum reputation — prevents permanent exclusion so every client
        retains a chance to be re-selected.
    ceiling : float
        Maximum reputation (typically ``1.0``).
    initial_reputation : float
        Reputation assigned to newly registered clients.
    penalty_factor : float
        Fraction of ``|G_i|`` subtracted from reputation when the update
        is actively harmful (``G_i < -θ``).  Set to ``0.0`` for the
        "no-change-on-invalid" variant from the thesis pseudo-code.
    """
    theta: float = 0.0               # validation-gain threshold
    gamma: float = 0.10              # reputation update factor
    decay_rate: float = 0.995        # per-round decay for non-participants
    floor: float = 0.05              # minimum reputation
    ceiling: float = 1.0             # maximum reputation
    initial_reputation: float = 0.5  # starting score
    penalty_factor: float = 0.0      # penalty multiplier for harmful updates


# ====================================================================== #
#  2.  LEDGER ENTRY — per-client record                                   #
# ====================================================================== #

@dataclass
class ClientLedgerEntry:
    """
    Full reputation record for a single client, including audit history.
    """
    client_id: str
    reputation: float = 0.5
    total_rounds_participated: int = 0
    total_valid_updates: int = 0
    total_invalid_updates: int = 0
    cumulative_gain: float = 0.0
    last_participated_round: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    # ---- serialisation ----------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "reputation": self.reputation,
            "total_rounds_participated": self.total_rounds_participated,
            "total_valid_updates": self.total_valid_updates,
            "total_invalid_updates": self.total_invalid_updates,
            "cumulative_gain": self.cumulative_gain,
            "last_participated_round": self.last_participated_round,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClientLedgerEntry":
        return cls(**d)


# ====================================================================== #
#  3.  CLIENT REPUTATION LEDGER (extended)                                #
# ====================================================================== #

class ClientReputationLedger:
    """
    Persistent, auditable reputation ledger for all federated clients.

    Implements the thesis pseudo-code:

    .. code-block:: text

        For each client i:
            G_i = validation_gain(u_i)
            if G_i > θ:  is_valid, delta_i = True,  G_i
            else:         is_valid, delta_i = False, 0
            if is_valid:  R_i = min(1, R_i + γ · delta_i)
            else:         R_i = R_i          # no change
            store(i, R_i)

    Additionally supports:

    * Optional **penalty** for harmful updates (``G_i < -θ``).
    * **Decay** for non-participating clients each round.
    * Full **history** / audit trail per client.
    * **JSON persistence** (save / load).
    * Backward-compatible bridge to the basic ``ReputationLedger`` from
      Part 1 via :meth:`as_basic_ledger`.

    Parameters
    ----------
    config : ReputationConfig
        All tuneable knobs.
    """

    def __init__(self, config: Optional[ReputationConfig] = None) -> None:
        self.config = config or ReputationConfig()
        self._entries: Dict[str, ClientLedgerEntry] = {}

    # ------------------------------------------------------------------ #
    #  Registration                                                       #
    # ------------------------------------------------------------------ #

    def register(self, client_id: str) -> None:
        """Register a new client with the default initial reputation."""
        if client_id not in self._entries:
            self._entries[client_id] = ClientLedgerEntry(
                client_id=client_id,
                reputation=self.config.initial_reputation,
            )
            logger.debug("Registered client %s (R=%.4f).",
                         client_id, self.config.initial_reputation)

    def register_many(self, client_ids: List[str]) -> None:
        for cid in client_ids:
            self.register(cid)

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def get(self, client_id: str) -> float:
        """Return the current reputation for *client_id*."""
        entry = self._entries.get(client_id)
        if entry is None:
            return self.config.initial_reputation
        return entry.reputation

    def get_entry(self, client_id: str) -> Optional[ClientLedgerEntry]:
        """Return the full ledger entry (or ``None``)."""
        return self._entries.get(client_id)

    def all_reputations(self) -> Dict[str, float]:
        """Return ``{client_id: reputation}`` for every registered client."""
        return {cid: e.reputation for cid, e in self._entries.items()}

    # ------------------------------------------------------------------ #
    #  Core update — implements the thesis pseudo-code                    #
    # ------------------------------------------------------------------ #

    def update_reputation(
        self,
        client_id: str,
        validation_gain: float,
        current_round: int,
        contribution_weight: Optional[float] = None,
    ) -> float:
        """
        Update a single client's reputation based on validation gain.

        Follows the pseudo-code:

        .. code-block:: text

            if G_i > θ:  R_i = min(ceiling, R_i + γ · G_i)
            else:        R_i = R_i                       # no change
            (optional)   if G_i < -θ:  R_i = max(floor, R_i - penalty · |G_i|)

        Parameters
        ----------
        client_id : str
        validation_gain : float
            ``G_i`` — the measured improvement on the server validation set.
        current_round : int
            Current federated round number (for the audit trail).
        contribution_weight : float | None
            ``c_i`` from Part 2 (stored in the audit trail, not used in
            the reputation formula itself).

        Returns
        -------
        new_reputation : float
        """
        self.register(client_id)                 # idempotent
        entry = self._entries[client_id]
        cfg = self.config
        old_rep = entry.reputation

        # --- 1. Determine validity ------------------------------------ #
        is_valid = validation_gain > cfg.theta
        delta = validation_gain if is_valid else 0.0

        # --- 2. Compute new reputation -------------------------------- #
        if is_valid:
            new_rep = min(cfg.ceiling, old_rep + cfg.gamma * delta)
        else:
            new_rep = old_rep                    # no change on invalid

        # --- 2b. Optional penalty for actively harmful updates -------- #
        if cfg.penalty_factor > 0 and validation_gain < -cfg.theta:
            penalty = cfg.penalty_factor * abs(validation_gain)
            new_rep = max(cfg.floor, new_rep - penalty)

        # --- 3. Clamp to [floor, ceiling] ----------------------------- #
        new_rep = float(np.clip(new_rep, cfg.floor, cfg.ceiling))

        # --- 4. Store ------------------------------------------------- #
        entry.reputation = new_rep
        entry.total_rounds_participated += 1
        entry.last_participated_round = current_round
        entry.cumulative_gain += validation_gain
        if is_valid:
            entry.total_valid_updates += 1
        else:
            entry.total_invalid_updates += 1

        # Audit trail
        entry.history.append({
            "round": current_round,
            "G_i": round(validation_gain, 6),
            "is_valid": is_valid,
            "delta": round(delta, 6),
            "R_old": round(old_rep, 6),
            "R_new": round(new_rep, 6),
            "c_i": round(contribution_weight, 6) if contribution_weight is not None else None,
            "ts": time.time(),
        })

        logger.debug(
            "Client %s: G_i=%.4f, valid=%s, R %.4f → %.4f",
            client_id, validation_gain, is_valid, old_rep, new_rep,
        )
        return new_rep

    # ------------------------------------------------------------------ #
    #  Batch update — process all clients in a round                      #
    # ------------------------------------------------------------------ #

    def update_round(
        self,
        gains: Dict[str, float],
        current_round: int,
        contribution_weights: Optional[Dict[str, float]] = None,
        participant_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Update reputations for an entire federated round.

        Parameters
        ----------
        gains : dict[str, float]
            ``{client_id: G_i}`` for each participating client.
        current_round : int
        contribution_weights : dict[str, float] | None
            ``{client_id: c_i}`` from Part 2 (for the audit trail).
        participant_ids : list[str] | None
            Explicit list of participants.  If ``None``, derived from
            ``gains.keys()``.

        Returns
        -------
        updated : dict[str, float]
            ``{client_id: new_reputation}`` for every registered client
            (including decayed non-participants).
        """
        participants = set(participant_ids or gains.keys())
        cw = contribution_weights or {}

        # Update participants
        for cid in participants:
            g = gains.get(cid, 0.0)
            c = cw.get(cid)
            self.update_reputation(cid, g, current_round, c)

        # Decay non-participants
        self._decay_non_participants(participants, current_round)

        return self.all_reputations()

    # ------------------------------------------------------------------ #
    #  Decay for idle clients                                             #
    # ------------------------------------------------------------------ #

    def _decay_non_participants(
        self,
        participants: set,
        current_round: int,
    ) -> None:
        """
        Apply multiplicative decay to clients that did **not** participate
        this round.  Keeps reputation ≥ floor.
        """
        cfg = self.config
        if cfg.decay_rate >= 1.0:
            return                               # decay disabled

        for cid, entry in self._entries.items():
            if cid in participants:
                continue
            old = entry.reputation
            new = max(cfg.floor, old * cfg.decay_rate)
            if abs(new - old) > 1e-9:
                entry.reputation = new
                entry.history.append({
                    "round": current_round,
                    "event": "decay",
                    "R_old": round(old, 6),
                    "R_new": round(new, 6),
                    "ts": time.time(),
                })
                logger.debug(
                    "Client %s decayed: R %.4f → %.4f (non-participant).",
                    cid, old, new,
                )

    # ------------------------------------------------------------------ #
    #  Querying & analytics                                               #
    # ------------------------------------------------------------------ #

    def ranked(self, descending: bool = True) -> List[Tuple[str, float]]:
        """Return ``(client_id, reputation)`` sorted by reputation."""
        items = list(self.all_reputations().items())
        items.sort(key=lambda x: x[1], reverse=descending)
        return items

    def statistics(self) -> Dict[str, Any]:
        """Aggregate statistics across all registered clients."""
        reps = [e.reputation for e in self._entries.values()]
        if not reps:
            return {}
        arr = np.array(reps)
        return {
            "num_clients": len(reps),
            "mean_reputation": float(arr.mean()),
            "std_reputation": float(arr.std()),
            "min_reputation": float(arr.min()),
            "max_reputation": float(arr.max()),
            "median_reputation": float(np.median(arr)),
        }

    def client_summary(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Return a human-readable summary dict for one client."""
        entry = self._entries.get(client_id)
        if entry is None:
            return None
        return {
            "client_id": entry.client_id,
            "reputation": entry.reputation,
            "rounds_participated": entry.total_rounds_participated,
            "valid_updates": entry.total_valid_updates,
            "invalid_updates": entry.total_invalid_updates,
            "cumulative_gain": entry.cumulative_gain,
            "last_participated_round": entry.last_participated_round,
            "history_length": len(entry.history),
        }

    # ------------------------------------------------------------------ #
    #  Bridge to Part 1 basic ReputationLedger                            #
    # ------------------------------------------------------------------ #

    def as_basic_ledger(self) -> ReputationLedger:
        """
        Export current reputations into a new ``ReputationLedger``
        (Part 1 format) so the client selector can consume them directly.
        """
        basic = ReputationLedger(
            default_reputation=self.config.initial_reputation,
        )
        for cid, entry in self._entries.items():
            basic.register(cid)
            # Overwrite internal score to match our ledger
            basic._scores[cid] = entry.reputation
        return basic

    def sync_from_basic_ledger(self, basic: ReputationLedger) -> None:
        """
        Import scores from an existing ``ReputationLedger`` (Part 1)
        into this extended ledger — useful when migrating mid-experiment.
        """
        for cid, score in basic.summary().items():
            self.register(cid)
            self._entries[cid].reputation = score

    # ------------------------------------------------------------------ #
    #  Persistence  (JSON)                                                #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Serialise the full ledger (including history) to a JSON file."""
        data = {
            "config": {
                "theta": self.config.theta,
                "gamma": self.config.gamma,
                "decay_rate": self.config.decay_rate,
                "floor": self.config.floor,
                "ceiling": self.config.ceiling,
                "initial_reputation": self.config.initial_reputation,
                "penalty_factor": self.config.penalty_factor,
            },
            "entries": {cid: e.to_dict() for cid, e in self._entries.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Ledger saved to %s (%d clients).", path, len(self._entries))

    @classmethod
    def load(cls, path: str | Path) -> "ClientReputationLedger":
        """Load a ledger from a previously saved JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = ReputationConfig(**raw["config"])
        ledger = cls(config=cfg)
        for cid, edict in raw["entries"].items():
            ledger._entries[cid] = ClientLedgerEntry.from_dict(edict)
        logger.info("Ledger loaded from %s (%d clients).", path, len(ledger._entries))
        return ledger


# ====================================================================== #
#  4.  INTEGRATION HELPER — plug into Parts 2 & 3                        #
# ====================================================================== #

def update_ledger_from_records(
    ledger: ClientReputationLedger,
    records,                        # List[ClientUpdateRecord] from Part 2
    current_round: int,
) -> Dict[str, float]:
    """
    Convenience function: extract ``G_i`` and ``c_i`` from Part 2's
    ``ClientUpdateRecord`` list and feed them into the extended ledger.

    Parameters
    ----------
    ledger : ClientReputationLedger
    records : list[ClientUpdateRecord]
        Output of ``UpdateValidator.validate_updates()``.
    current_round : int

    Returns
    -------
    updated : dict[str, float]
    """
    gains: Dict[str, float] = {}
    cws: Dict[str, float] = {}
    for rec in records:
        gains[rec.client_id] = rec.validation_gain
        cws[rec.client_id] = rec.contribution_weight
    return ledger.update_round(gains, current_round, contribution_weights=cws)


# ====================================================================== #
#  DEMO / SMOKE-TEST                                                      #
# ====================================================================== #

if __name__ == "__main__":
    print("\n===  Client Reputation & Ledger — Demo  ===\n")

    np.random.seed(42)

    NUM_CLIENTS = 8
    NUM_ROUNDS = 10

    # ---- Configuration ----------------------------------------------- #
    cfg = ReputationConfig(
        theta=0.0,                    # any positive gain counts
        gamma=0.10,                   # reputation update factor
        decay_rate=0.99,              # 1 % decay per idle round
        floor=0.05,                   # minimum reputation
        ceiling=1.0,
        initial_reputation=0.50,
        penalty_factor=0.05,          # small penalty for harmful updates
    )

    ledger = ClientReputationLedger(config=cfg)
    client_ids = [f"client_{i:02d}" for i in range(NUM_CLIENTS)]
    ledger.register_many(client_ids)

    print(f"Config: θ={cfg.theta}, γ={cfg.gamma}, decay={cfg.decay_rate}, "
          f"floor={cfg.floor}, penalty={cfg.penalty_factor}")
    print(f"Registered {NUM_CLIENTS} clients, initial R={cfg.initial_reputation}\n")

    # ---- Simulate rounds --------------------------------------------- #
    for rnd in range(1, NUM_ROUNDS + 1):
        # Each round, randomly select ~60 % of clients
        num_selected = max(2, int(0.6 * NUM_CLIENTS))
        participants = list(np.random.choice(client_ids, size=num_selected, replace=False))

        # Simulate validation gains — mostly small positive, some negative
        gains: Dict[str, float] = {}
        for cid in participants:
            g = np.random.normal(loc=0.02, scale=0.04)   # mean 2 %, σ 4 %
            gains[cid] = float(g)

        # Simulate contribution weights (from Part 2)
        cws = {cid: float(np.random.uniform(0.3, 1.0)) for cid in participants}

        updated = ledger.update_round(
            gains=gains,
            current_round=rnd,
            contribution_weights=cws,
            participant_ids=participants,
        )

        # Print round summary
        print(f"--- Round {rnd} ---  participants: {participants}")
        print(f"  Gains:  { {k: f'{v:+.4f}' for k, v in gains.items()} }")
        ranked = ledger.ranked()
        print(f"  Reputations: { {k: f'{v:.4f}' for k, v in ranked} }\n")

    # ---- Final statistics -------------------------------------------- #
    print("=" * 50)
    print("Final Ledger Statistics:")
    stats = ledger.statistics()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nPer-client summaries:")
    for cid in client_ids:
        s = ledger.client_summary(cid)
        print(f"  {cid}: R={s['reputation']:.4f}, "
              f"participated={s['rounds_participated']}, "
              f"valid={s['valid_updates']}, invalid={s['invalid_updates']}, "
              f"cumG={s['cumulative_gain']:+.4f}")

    # ---- Persistence round-trip -------------------------------------- #
    save_path = "demo_ledger.json"
    ledger.save(save_path)
    loaded = ClientReputationLedger.load(save_path)
    assert loaded.all_reputations() == ledger.all_reputations(), "Round-trip mismatch!"
    print(f"\nJSON round-trip OK ({save_path})")

    # ---- Bridge to Part 1 -------------------------------------------- #
    basic = ledger.as_basic_ledger()
    print(f"\nBasic ReputationLedger bridge: {basic.summary()}")

    # Clean up demo file
    Path(save_path).unlink(missing_ok=True)

    print("\nDone.")
