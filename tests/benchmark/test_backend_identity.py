from __future__ import annotations

import pytest

from src.benchmark import backend_identity
from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_SCHEMA,
    StrictJsonError,
    normalize_known_backend_descriptor,
)
from src.benchmark.muscriptor_model import MuscriptorModelLock, derive_muscriptor_model_id

MUSCRIPTOR_BACKEND_ID = "muscriptor-v0.3.0-drums-v1"
MUSCRIPTOR_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
MUSCRIPTOR_MODEL_ID = "muscriptor-medium-0123456789ab-fedcba987654"
MUSCRIPTOR_METADATA_SCHEMA = "muscriptor-note-start-metadata-v1"
MUSCRIPTOR_OUTPUT_SPACE = "muscriptor-drums-midi128-v1"
MUSCRIPTOR_TRAINING_DATA = "muscriptor-training-data-v0.3.0"
MUSCRIPTOR_COMMIT = "d73147e75e5b9b0c0a79ebe154587db4fd603e0c"


def _oaf_payload() -> dict[str, str]:
    return {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": OAF_BACKEND_ID,
        "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
    }


def _muscriptor_payload() -> dict[str, str]:
    return {
        "architecture_id": "muscriptor-transformer-v0.3.0",
        "backend_id": MUSCRIPTOR_BACKEND_ID,
        "descriptor_schema": MUSCRIPTOR_DESCRIPTOR_SCHEMA,
        "model_id": MUSCRIPTOR_MODEL_ID,
        "native_metadata_schema_id": MUSCRIPTOR_METADATA_SCHEMA,
        "native_output_space_id": MUSCRIPTOR_OUTPUT_SPACE,
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": MUSCRIPTOR_TRAINING_DATA,
        "upstream_source_commit": MUSCRIPTOR_COMMIT,
    }


def test_oaf_descriptor_still_requires_every_frozen_identity() -> None:
    for field in (
        "architecture_id",
        "backend_id",
        "descriptor_schema",
        "model_id",
        "native_metadata_schema_id",
        "native_output_space_id",
        "prediction_schema",
        "training_data_map_id",
    ):
        payload = _oaf_payload()
        payload[field] = "other"
        with pytest.raises(StrictJsonError, match=field if field != "backend_id" else "backend_id"):
            normalize_known_backend_descriptor(payload)


def test_muscriptor_descriptor_accepts_only_its_frozen_family_shape() -> None:
    assert backend_identity.MUSCRIPTOR_BACKEND_ID == MUSCRIPTOR_BACKEND_ID
    assert backend_identity.MUSCRIPTOR_DESCRIPTOR_SCHEMA == MUSCRIPTOR_DESCRIPTOR_SCHEMA
    assert backend_identity.MUSCRIPTOR_MODEL_ID_RE.fullmatch(MUSCRIPTOR_MODEL_ID)
    assert normalize_known_backend_descriptor(_muscriptor_payload()) == _muscriptor_payload()

    missing = _muscriptor_payload()
    missing.pop("architecture_id")
    with pytest.raises(StrictJsonError, match="exact key set"):
        normalize_known_backend_descriptor(missing)

    extra = _muscriptor_payload()
    extra["unexpected"] = "value"
    with pytest.raises(StrictJsonError, match="exact key set"):
        normalize_known_backend_descriptor(extra)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architecture_id", "magenta-oaf-model-tpu-drums-v1"),
        ("native_metadata_schema_id", "magenta-oaf-native-metadata-v1"),
        ("native_output_space_id", "magenta-oaf-midi88-a0-v1"),
        ("prediction_schema", "other/v1"),
        ("training_data_map_id", "other/v1"),
    ],
)
def test_muscriptor_descriptor_rejects_mixed_or_unfrozen_identities(
    field: str,
    value: str,
) -> None:
    payload = _muscriptor_payload()
    payload[field] = value

    with pytest.raises(StrictJsonError):
        normalize_known_backend_descriptor(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {**_oaf_payload(), "backend_id": MUSCRIPTOR_BACKEND_ID},
        {**_muscriptor_payload(), "backend_id": OAF_BACKEND_ID},
    ],
)
def test_mixed_oaf_and_muscriptor_identities_are_rejected(
    payload: dict[str, str],
) -> None:
    with pytest.raises(StrictJsonError):
        normalize_known_backend_descriptor(payload)


@pytest.mark.parametrize(
    "model_id",
    [
        "muscriptor-large-0123456789ab-fedcba987654",
        "muscriptor-medium-0123456789a-fedcba987654",
        "muscriptor-medium-0123456789ab-fedcba98765",
        "muscriptor-medium-0123456789AB-fedcba987654",
    ],
)
def test_muscriptor_model_id_requires_the_frozen_variant_and_hex_lengths(
    model_id: str,
) -> None:
    payload = _muscriptor_payload()
    payload["model_id"] = model_id
    with pytest.raises(StrictJsonError):
        normalize_known_backend_descriptor(payload)


def _model_lock() -> MuscriptorModelLock:
    revision = "a" * 40
    checkpoint_sha256 = "b" * 64
    return MuscriptorModelLock(
        package_name="muscriptor",
        package_version="0.3.0",
        upstream_source_commit=MUSCRIPTOR_COMMIT,
        code_license="MIT",
        weight_license="CC BY-NC 4.0",
        checkpoint_variant="medium",
        checkpoint_repo_id="MuScriptor/muscriptor-medium",
        checkpoint_revision=revision,
        checkpoint_filename="model.safetensors",
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_byte_length=1,
        checkpoint_config_filename="config.json",
        checkpoint_config_sha256="c" * 64,
        checkpoint_config_byte_length=1,
        model_id=f"muscriptor-medium-{revision[:12]}-{checkpoint_sha256[:12]}",
        device="cpu",
        dtype="float32",
        input_sample_rate_hz=16000,
        chunk_duration_sec=5.0,
        use_sampling=False,
        temperature=1.0,
        cfg_coef=1.0,
        instruments=("drums",),
        batch_size=1,
        no_eos_is_ok=True,
        beam_size=1,
        prelude_forcing=True,
        native_output_space_id=MUSCRIPTOR_OUTPUT_SPACE,
        native_metadata_schema_id=MUSCRIPTOR_METADATA_SCHEMA,
        training_data_map_id=MUSCRIPTOR_TRAINING_DATA,
    )


def test_runner_helper_derives_exact_model_id_from_the_model_lock() -> None:
    helper = getattr(backend_identity, "expected_muscriptor_model_id")
    lock = _model_lock()

    assert helper(lock) == derive_muscriptor_model_id(lock)
    assert helper(lock) == "muscriptor-medium-aaaaaaaaaaaa-bbbbbbbbbbbb"
