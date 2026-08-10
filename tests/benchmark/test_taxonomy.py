from typing import get_args

from src.benchmark.backend_identity import OAF_BACKEND_ID
from src.benchmark.prediction_artifact import OAF_GROUP_IDS
from src.benchmark.taxonomy import (
    DETAILED_TO_COMMON,
    DTX_LANE_MAP,
    OAF_PREDICTION_MAP,
    ClassMapping,
    DetailedDrumClass,
    project_to_common,
)


def test_detailed_to_common_projection_is_total() -> None:
    assert DETAILED_TO_COMMON == {
        "kick": "kick",
        "snare": "snare",
        "closed_hihat": "hihat",
        "open_hihat": "hihat",
        "crash": "crash",
        "ride": "ride",
        "high_tom": "tom",
        "low_or_floor_tom": "tom",
    }
    assert set(DETAILED_TO_COMMON) == set(get_args(DetailedDrumClass))


def test_dtx_lane_map_uses_frozen_detailed_and_common_classes() -> None:
    expected = {
        "11": ("closed_hihat", "hihat"),
        "12": ("snare", "snare"),
        "13": ("kick", "kick"),
        "14": ("high_tom", "tom"),
        "15": ("low_or_floor_tom", "tom"),
        "16": ("crash", "crash"),
        "17": ("low_or_floor_tom", "tom"),
        "18": ("open_hihat", "hihat"),
        "19": ("ride", "ride"),
        "1A": ("crash", "crash"),
        "1B": ("closed_hihat", "hihat"),
        "1C": ("kick", "kick"),
    }

    assert {
        lane_id: (mapping.canonical_class, mapping.common_class)
        for lane_id, mapping in DTX_LANE_MAP.items()
    } == expected
    for mapping in DTX_LANE_MAP.values():
        if mapping.canonical_class is not None:
            assert mapping.common_class == project_to_common(mapping.canonical_class)


def test_oaf_prediction_map_binds_to_locked_vocabulary() -> None:
    assert set(OAF_PREDICTION_MAP.classes) == OAF_GROUP_IDS
    assert OAF_PREDICTION_MAP.model_id == OAF_BACKEND_ID
    assert OAF_PREDICTION_MAP.classes["hihat"] == ClassMapping(None, "hihat")
    assert OAF_PREDICTION_MAP.classes["toms"] == ClassMapping(None, "tom")
    assert OAF_PREDICTION_MAP.classes["sticks"] == ClassMapping(None, None)
