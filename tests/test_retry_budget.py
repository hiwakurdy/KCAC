from __future__ import annotations

import pytest

from pipeline.budget import BudgetTracker
from pipeline.models import BudgetEvent
from pipeline.retry import ConsecutiveFailureGuard, retry_call


def test_retry_call_eventually_succeeds() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise ValueError("not yet")
        return "ok"

    assert retry_call(flaky, attempts=2, initial_delay_seconds=0) == "ok"


def test_failure_guard_stops_after_limit() -> None:
    guard = ConsecutiveFailureGuard(max_failures=2)
    guard.record_failure()
    with pytest.raises(RuntimeError):
        guard.record_failure()


def test_budget_tracker_totals_by_book() -> None:
    tracker = BudgetTracker()
    tracker.add(BudgetEvent("claude", "b1", "p1", 1, 0.25, "test"))
    tracker.add(BudgetEvent("claude", "b1", "p2", 1, 0.25, "test"))
    assert tracker.total_usd() == 0.5
    assert tracker.by_book() == {"b1": 0.5}
