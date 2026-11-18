"""
inventory_policy.py
===================
Implements the continuous-review (s, Q) replenishment policy.

Policy definition
-----------------
Review: continuous (checked every day).
Trigger: order placed when inventory position <= reorder_point (s).
Quantity: fixed order quantity Q.

Inventory position = on-hand + on-order - backorders.

Because we use a lost-sales model (not backorders), inventory position
simplifies to: on-hand + on-order.

Lead time
---------
Lead time is drawn from Discrete Uniform(lt_min, lt_max).  Replenishments
placed on day t arrive at the start of day t + lead_time (i.e. they are
available to fulfil demand on that arrival day).

Supplier disruptions
--------------------
When disruption_prob > 0, there is a daily probability of a disruption event.
During a disruption, no new replenishment orders can be placed and in-transit
orders may be delayed by an additional disruption_duration days.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PolicyParameters:
    """Container for all (s, Q) policy parameters.

    Attributes
    ----------
    reorder_point : int
        Reorder point s.  An order is triggered when inventory_position <= s.
    order_quantity : int
        Fixed replenishment batch size Q (units per order).
    lt_min : int
        Minimum supplier lead time in days.
    lt_max : int
        Maximum supplier lead time in days.
    disruption_prob : float
        Daily probability of a supplier disruption event (0 ≤ p ≤ 1).
    disruption_duration_max : int
        Maximum duration of a disruption (days); drawn Uniform(1, max).
    safety_stock_extra : int
        Additional buffer units added to opening inventory (applied once
        at simulation start).  Used in the safety-stock scenario.
    """
    reorder_point: int
    order_quantity: int
    lt_min: int
    lt_max: int
    disruption_prob: float = 0.0
    disruption_duration_max: int = 0
    safety_stock_extra: int = 0

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.reorder_point < 0:
            raise ValueError("reorder_point must be >= 0.")
        if self.order_quantity <= 0:
            raise ValueError("order_quantity must be > 0.")
        if self.lt_min < 1 or self.lt_max < self.lt_min:
            raise ValueError("Lead times must satisfy 1 <= lt_min <= lt_max.")
        if not (0.0 <= self.disruption_prob <= 1.0):
            raise ValueError("disruption_prob must be in [0, 1].")
        if self.disruption_duration_max < 0:
            raise ValueError("disruption_duration_max must be >= 0.")
        if self.safety_stock_extra < 0:
            raise ValueError("safety_stock_extra must be >= 0.")


@dataclass
class PendingOrder:
    """Represents a replenishment order that is in transit.

    Attributes
    ----------
    quantity : int
        Number of units ordered.
    arrival_day : int
        Simulation day index on which units arrive (0-indexed).
    """
    quantity: int
    arrival_day: int


class InventoryPolicy:
    """Implements the continuous-review (s, Q) inventory control policy.

    Parameters
    ----------
    params : PolicyParameters
    rng    : numpy random Generator (passed in; ensures reproducibility)
    """

    def __init__(self, params: PolicyParameters, rng) -> None:
        self._params = params
        self._rng = rng

    # ------------------------------------------------------------------
    # Lead-time sampling
    # ------------------------------------------------------------------

    def sample_lead_time(self) -> int:
        """Sample a supplier lead time in days.

        Returns
        -------
        int in [lt_min, lt_max] drawn from Discrete Uniform.
        """
        return int(self._rng.integers(self._params.lt_min, self._params.lt_max + 1))

    # ------------------------------------------------------------------
    # Disruption sampling
    # ------------------------------------------------------------------

    def disruption_occurs(self) -> bool:
        """Return True if a supplier disruption event occurs today."""
        if self._params.disruption_prob <= 0.0:
            return False
        return bool(self._rng.random() < self._params.disruption_prob)

    def sample_disruption_duration(self) -> int:
        """Sample disruption duration in days (at least 1 if max > 0)."""
        if self._params.disruption_duration_max <= 0:
            return 0
        return int(self._rng.integers(1, self._params.disruption_duration_max + 1))

    # ------------------------------------------------------------------
    # Replenishment decision
    # ------------------------------------------------------------------

    def should_reorder(
        self,
        inventory_position: int,
        disruption_active: bool,
    ) -> bool:
        """Decide whether to place a replenishment order.

        An order is placed when:
        - inventory_position <= reorder_point, AND
        - no supplier disruption is currently active.

        Parameters
        ----------
        inventory_position : on-hand + on-order (units)
        disruption_active  : True if the supplier is currently disrupted

        Returns
        -------
        bool
        """
        if disruption_active:
            return False
        return inventory_position <= self._params.reorder_point

    def create_order(self, current_day: int) -> PendingOrder:
        """Create a new PendingOrder with the fixed order quantity.

        Parameters
        ----------
        current_day : zero-indexed simulation day index

        Returns
        -------
        PendingOrder with arrival_day = current_day + lead_time
        """
        lt = self.sample_lead_time()
        return PendingOrder(
            quantity=self._params.order_quantity,
            arrival_day=current_day + lt,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> PolicyParameters:
        """Return the policy parameters (read-only)."""
        return self._params
