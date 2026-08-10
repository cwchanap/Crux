from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

from src.benchmark.backend_identity import OAF_BACKEND_ID

DetailedDrumClass: TypeAlias = Literal[
    "kick",
    "snare",
    "closed_hihat",
    "open_hihat",
    "crash",
    "ride",
    "high_tom",
    "low_or_floor_tom",
]

CommonDrumClass: TypeAlias = Literal[
    "kick",
    "snare",
    "hihat",
    "crash",
    "ride",
    "tom",
]

TAXONOMY_VERSION = "crux.drum-taxonomy/v1"
DTX_LANE_MAP_VERSION = "crux.dtx-lane-map/v1"
OAF_PREDICTION_MAP_ID = "crux.prediction-map/oaf-egmd-8hit-v1"

DETAILED_TO_COMMON: Mapping[DetailedDrumClass, CommonDrumClass] = MappingProxyType(
    {
        "kick": "kick",
        "snare": "snare",
        "closed_hihat": "hihat",
        "open_hihat": "hihat",
        "crash": "crash",
        "ride": "ride",
        "high_tom": "tom",
        "low_or_floor_tom": "tom",
    }
)


def project_to_common(detailed: DetailedDrumClass) -> CommonDrumClass:
    return DETAILED_TO_COMMON[detailed]


@dataclass(frozen=True)
class ClassMapping:
    canonical_class: DetailedDrumClass | None
    common_class: CommonDrumClass | None


@dataclass(frozen=True)
class PredictionMap:
    map_id: str
    model_id: str
    native_output_space_id: str
    classes: Mapping[str, ClassMapping]


DTX_LANE_MAP: Mapping[str, ClassMapping] = MappingProxyType(
    {
        "11": ClassMapping("closed_hihat", "hihat"),
        "12": ClassMapping("snare", "snare"),
        "13": ClassMapping("kick", "kick"),
        "14": ClassMapping("high_tom", "tom"),
        "15": ClassMapping("low_or_floor_tom", "tom"),
        "16": ClassMapping("crash", "crash"),
        "17": ClassMapping("low_or_floor_tom", "tom"),
        "18": ClassMapping("open_hihat", "hihat"),
        "19": ClassMapping("ride", "ride"),
        "1A": ClassMapping("crash", "crash"),
        "1B": ClassMapping("closed_hihat", "hihat"),
        "1C": ClassMapping("kick", "kick"),
    }
)

DRUM_LANE_IDS: frozenset[str] = frozenset(DTX_LANE_MAP)
IGNORED_NON_DRUM_LANES: frozenset[str] = frozenset()

OAF_PREDICTION_MAP = PredictionMap(
    map_id=OAF_PREDICTION_MAP_ID,
    model_id=OAF_BACKEND_ID,
    native_output_space_id="magenta-oaf-midi88-a0-v1",
    classes=MappingProxyType(
        {
            "kick": ClassMapping("kick", "kick"),
            "snare": ClassMapping("snare", "snare"),
            "toms": ClassMapping(None, "tom"),
            "hihat": ClassMapping(None, "hihat"),
            "ride": ClassMapping("ride", "ride"),
            "ride_bell": ClassMapping("ride", "ride"),
            "crash": ClassMapping("crash", "crash"),
            "sticks": ClassMapping(None, None),
        }
    ),
)
