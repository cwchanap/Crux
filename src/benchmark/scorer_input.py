from __future__ import annotations

from src.benchmark.backends import NativeEvent
from src.benchmark.prediction_artifact import read_prediction_artifact


class CanonicalMappingRequired(ValueError):
    pass


def read_scorer_events(content: bytes) -> tuple[NativeEvent, ...]:
    read_prediction_artifact(content)
    raise CanonicalMappingRequired("canonical_mapping_required")
