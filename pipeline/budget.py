from __future__ import annotations

from dataclasses import dataclass, field

from .models import BudgetEvent


@dataclass(slots=True)
class BudgetTracker:
    events: list[BudgetEvent] = field(default_factory=list)

    def add(self, event: BudgetEvent) -> None:
        self.events.append(event)

    def total_usd(self) -> float:
        return round(sum(event.estimated_usd for event in self.events), 6)

    def by_book(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for event in self.events:
            totals[event.book_id] = totals.get(event.book_id, 0.0) + event.estimated_usd
        return {key: round(value, 6) for key, value in totals.items()}

    def to_json(self) -> dict[str, object]:
        return {
            "total_usd": self.total_usd(),
            "by_book": self.by_book(),
            "events": [event.to_json() for event in self.events],
        }
