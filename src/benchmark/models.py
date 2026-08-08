from __future__ import annotations

from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any


@dataclass(frozen=True)
class DtxEvent:
    chart_id: str
    measure: int
    position: float
    lane_id: str
    note_id: str
    source_order: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_id", self.lane_id.upper())
        object.__setattr__(self, "note_id", self.note_id.upper())


@total_ordering
@dataclass(frozen=True)
class BenchmarkEvent:
    chart_id: str
    time_sec: float
    canonical_class: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict, hash=False)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, BenchmarkEvent):
            return NotImplemented
        return (self.time_sec, self.canonical_class) < (other.time_sec, other.canonical_class)


@dataclass(frozen=True)
class MatchResult:
    ground_truth: BenchmarkEvent
    prediction: BenchmarkEvent
    timing_error_sec: float

    @property
    def absolute_error_sec(self) -> float:
        return abs(self.timing_error_sec)


@dataclass(frozen=True)
class ScoreSummary:
    true_positives: int
    false_positives: int
    false_negatives: int
    median_abs_error_sec: float | None = None
    p95_abs_error_sec: float | None = None
    offset_sec: float = 0.0

    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        precision = self.precision
        recall = self.recall
        if precision is None and recall is None:
            return None
        p = precision if precision is not None else 0.0
        r = recall if recall is not None else 0.0
        denominator = p + r
        return 2 * p * r / denominator if denominator else 0.0
