"""Reservation accounting includes failed attempts; cache reuse consumes no new call."""
from dataclasses import asdict, dataclass


@dataclass
class BudgetManager:
    views: int = 0
    visual_tokens: int = 0
    tiles: int = 0
    elapsed: float = 0.0

    def rejection(self, view, config):
        tokens = view.plan.estimated_visual_tokens
        if self.elapsed >= config.max_latency_per_table:
            return "latency_budget_exhausted"
        if self.views >= config.max_views_per_table:
            return "view_budget_exhausted"
        if view.channel_id == "tiles" and self.tiles >= config.max_tiles_per_table:
            return "tile_budget_exhausted"
        if tokens > config.max_visual_tokens_per_view:
            return "per_view_visual_token_budget_exhausted"
        if self.visual_tokens + tokens > config.max_total_visual_tokens:
            return "total_visual_token_budget_exhausted"
        return None

    def reserve(self, view, config):
        reason = self.rejection(view, config)
        if reason:
            raise ValueError(reason)
        self.views += 1
        self.visual_tokens += view.plan.estimated_visual_tokens
        self.tiles += view.channel_id == "tiles"

    def as_dict(self):
        return asdict(self)
