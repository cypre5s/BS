"""Conservative image-scale budgets for the official processor."""

from dataclasses import dataclass

from xiaomi_runtime.processor import ProcessorLimits


@dataclass(frozen=True)
class ResolutionPolicy:
    medium_scale: float = 1.5
    high_scale: float = 2.0
    medium_max_visual_tokens: int = 2048
    high_max_visual_tokens: int = 4096

    def views(self, size: tuple[int, int], limits: ProcessorLimits) -> dict[str, tuple[int, int]]:
        native = limits.native_size(size)
        native_pixels = native[0] * native[1]
        result = {}
        for name, scale, budget in (
            ("medium", self.medium_scale, self.medium_max_visual_tokens),
            ("high", self.high_scale, self.high_max_visual_tokens),
        ):
            if scale < 1 or budget < 1:
                raise ValueError("Enhanced scales and token budgets must be positive")
            max_pixels = min(budget * limits.factor**2, limits.max_pixels)
            if native_pixels >= max_pixels:
                result[name] = native
                continue
            requested = (round(native[0] * scale), round(native[1] * scale))
            result[name] = limits.native_size(requested, max_pixels=max_pixels)
        return result
