"""Optional inference guards; the historical runtime has no guards by default."""

from collections.abc import Callable
from time import perf_counter
from typing import Any


class BudgetExceeded(ValueError):
    """A call was refused before generation, without spending its visual tokens."""


class DeadlineExceeded(TimeoutError):
    """A deadline was reached before generation could start."""


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise DeadlineExceeded("Inference deadline reached")


class DeadlineStop:
    """Cooperative stop between decoding steps, never a CUDA preemption claim."""

    def __init__(self, deadline: float, clock: Callable[[], float] = perf_counter):
        self.deadline = deadline
        self.clock = clock
        self.expired = False

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        self.expired = self.clock() >= self.deadline
        return self.expired
