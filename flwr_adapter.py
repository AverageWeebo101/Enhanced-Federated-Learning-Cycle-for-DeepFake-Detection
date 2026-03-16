"""
Deprecated adapter shim.

The project has been migrated to native Flower execution via
flwr_federated_cycle.py. This module remains only to avoid immediate
import errors in stale notebook cells.
"""

from __future__ import annotations


class _DeprecatedCompat:
    def __getattr__(self, name):
        raise RuntimeError(
            "flwr_adapter.tff_compat is deprecated. "
            "Use flwr_federated_cycle directly."
        )


tff_compat = _DeprecatedCompat()


__all__ = ["tff_compat"]
