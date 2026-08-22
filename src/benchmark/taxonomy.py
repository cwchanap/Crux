from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

from src.benchmark.backend_identity import IDM_BACKEND_ID, MUSCRIPTOR_BACKEND_ID, OAF_BACKEND_ID

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
MUSCRIPTOR_PREDICTION_MAP_ID = "crux.prediction-map/muscriptor-drums-v1"
IDM_PREDICTION_MAP_ID = "crux.prediction-map/idm-44-train-kits-v1"

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
    backend_id: str
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
# DTXMania's channel enum identifies hexadecimal channel 0x54 as ``Movie`` and
# 0xC2 as ``BeatLineDisplay``; neither is a playable drum lane.  The HPA-323
# corpus audit observed both channels, so retain them as explicit non-drum
# exclusions rather than treating them as unresolved taxonomy gaps.
IGNORED_NON_DRUM_LANES: frozenset[str] = frozenset({"54", "C2"})

OAF_PREDICTION_MAP = PredictionMap(
    map_id=OAF_PREDICTION_MAP_ID,
    backend_id=OAF_BACKEND_ID,
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

MUSCRIPTOR_PREDICTION_MAP = PredictionMap(
    map_id=MUSCRIPTOR_PREDICTION_MAP_ID,
    backend_id=MUSCRIPTOR_BACKEND_ID,
    native_output_space_id="muscriptor-drums-midi128-v1",
    classes=MappingProxyType(
        {
            "35": ClassMapping("kick", "kick"),
            "36": ClassMapping("kick", "kick"),
            "38": ClassMapping("snare", "snare"),
            "40": ClassMapping("snare", "snare"),
            "41": ClassMapping("low_or_floor_tom", "tom"),
            "43": ClassMapping("low_or_floor_tom", "tom"),
            "45": ClassMapping("low_or_floor_tom", "tom"),
            "47": ClassMapping("low_or_floor_tom", "tom"),
            "48": ClassMapping("high_tom", "tom"),
            "50": ClassMapping("high_tom", "tom"),
            "42": ClassMapping("closed_hihat", "hihat"),
            "44": ClassMapping("closed_hihat", "hihat"),
            "46": ClassMapping("open_hihat", "hihat"),
            "49": ClassMapping("crash", "crash"),
            "57": ClassMapping("crash", "crash"),
            "51": ClassMapping("ride", "ride"),
            "53": ClassMapping("ride", "ride"),
            "59": ClassMapping("ride", "ride"),
        }
    ),
)

IDM_PREDICTION_MAP = PredictionMap(
    map_id=IDM_PREDICTION_MAP_ID,
    backend_id=IDM_BACKEND_ID,
    native_output_space_id="idm-44-train-kits-9class-v1",
    classes=MappingProxyType(
        {
            "KD": ClassMapping("kick", "kick"),
            "SD": ClassMapping("snare", "snare"),
            "HH_CHH": ClassMapping("closed_hihat", "hihat"),
            "HH_OHH": ClassMapping("open_hihat", "hihat"),
            "CY_CR": ClassMapping("crash", "crash"),
            "CY_RD": ClassMapping("ride", "ride"),
            "TT_HMT": ClassMapping("high_tom", "tom"),
            "TT_LMT": ClassMapping(None, "tom"),
            "TT_HFT": ClassMapping("low_or_floor_tom", "tom"),
        }
    ),
)
