"""Explainable scale selection and cumulative resource admission."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

from .complexity import ComplexityFeatures
from .config import SAMRConfig
from .scales import ScalePlan


@dataclass
class RoutingDecision:
    policy: str
    level: str
    desired_scales: tuple[float, ...]
    reason: str
    events: list[dict] = field(default_factory=list)


def route(config: SAMRConfig, features: ComplexityFeatures) -> RoutingDecision:
    if config.mode != "adaptive":
        desired = config.scales
        reason = "explicit_development_scale_configuration"
    else:
        desired = config.scales[:{"LOW": 1, "MEDIUM": 2, "HIGH": 3}[features.level]]
        reason = f"predetermined_image_score_{features.score:.6f}_{features.level}"
    return RoutingDecision(config.routing_policy, features.level, desired, reason)


@dataclass
class BudgetLedger:
    config: SAMRConfig
    clock: Callable[[], float] = perf_counter
    started: float = field(init=False)
    spent_tokens: int = 0
    started_scales: int = 0
    processed_sizes: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.started = self.clock()

    @property
    def deadline(self) -> float:
        return self.started + self.config.max_latency_budget

    @property
    def remaining_tokens(self) -> int:
        return self.config.max_visual_tokens - self.spent_tokens

    def admit(self, plan: ScalePlan) -> str:
        if self.clock() >= self.deadline:
            return "latency_budget_exhausted"
        if self.started_scales >= self.config.max_scales_per_table:
            return "scale_count_budget_exhausted"
        if plan.estimated_visual_tokens > self.remaining_tokens:
            return "visual_token_budget_exhausted"
        if plan.processed_size in self.processed_sizes:
            return "duplicate_native_resolution"
        return "admitted"

    def reserve(self, plan: ScalePlan) -> None:
        reason = self.admit(plan)
        if reason != "admitted":
            raise ValueError(reason)
        # Reservation also charges failed/aborted calls conservatively.
        self.spent_tokens += plan.estimated_visual_tokens
        self.started_scales += 1
        self.processed_sizes.add(plan.processed_size)

    def as_dict(self) -> dict:
        elapsed = self.clock() - self.started
        return {"reserved_visual_tokens": self.spent_tokens, "started_scales": self.started_scales,
                "elapsed_seconds": elapsed, "max_visual_tokens": self.config.max_visual_tokens,
                "max_scales_per_table": self.config.max_scales_per_table,
                "max_latency_budget": self.config.max_latency_budget,
                "deadline_exceeded": elapsed > self.config.max_latency_budget,
                "deadline_overshoot_seconds": max(0.0, elapsed - self.config.max_latency_budget),
                "latency_semantics": "cooperative_decode_deadline; nonpreemptible_forward_may_overshoot"}
