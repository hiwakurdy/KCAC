from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class ConsecutiveFailureGuard:
    def __init__(self, max_failures: int = 5) -> None:
        self.max_failures = max_failures
        self.failures = 0

    def record_success(self) -> None:
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.max_failures:
            raise RuntimeError(f"Stopping after {self.failures} consecutive engine failures")


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    initial_delay_seconds: float = 1.0,
    multiplier: float = 2.0,
) -> T:
    delay = initial_delay_seconds
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay)
            delay *= multiplier
    assert last_error is not None
    raise last_error
