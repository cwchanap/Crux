from __future__ import annotations

# Exact key-set fixtures intentionally mirror the external v1 schemas.
# pylint: disable=duplicate-code,too-many-lines
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from src.benchmark import backend_publication, backend_reports
from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex, strict_json_loads
from src.benchmark.backend_reports import (
    ExecutionReport,
    LegacyScoreReport,
    OperationalReportPublicationError,
    ReportValidationError,
    VerificationReport,
    derive_item_id,
    publish_operational_report,
)

FIXED_UTC = datetime(2026, 7, 27, 1, 2, 3, 456789, tzinfo=UTC)
FIXED_UUID = UUID("12345678-1234-4678-9234-567812345678")
OAF_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"
VERIFICATION_KEYS = {
    "schema",
    "report_type",
    "run_id",
    "started_at",
    "finished_at",
    "status",
    "exit_code",
    "descriptor",
    "descriptor_sha256",
    "backend_lock_sha256",
    "runtime_lock_sha256",
    "parameter_lock_sha256",
    "seal_evidence_sha256",
    "execution_attestation",
    "tensor_coverage",
    "smoke",
    "artifacts",
    "errors",
}
EXECUTION_KEYS = {
    "schema",
    "report_type",
    "run_id",
    "started_at",
    "finished_at",
    "status",
    "exit_code",
    "descriptor",
    "descriptor_sha256",
    "backend_lock_sha256",
    "runtime_lock_sha256",
    "parameter_lock_sha256",
    "seal_evidence_sha256",
    "execution_attestation",
    "verification_report",
    "items",
    "errors",
}
LEGACY_KEYS = {
    "schema",
    "report_type",
    "run_id",
    "started_at",
    "finished_at",
    "workflow_mode",
    "backend_id",
    "backend_validation_status",
    "status",
    "exit_code",
    "score_report",
    "errors",
}
OAF_DESCRIPTOR = {
    "architecture_id": "magenta-oaf-model-tpu-drums-v1",
    "backend_id": OAF_BACKEND_ID,
    "backend_lock_sha256": "b" * 64,
    "descriptor_schema": "crux.transcription-backend-descriptor/v1",
    "model_artifact_set_sha256": "7" * 64,
    "model_id": "magenta-egmd-ckpt-569400-v1",
    "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
    "native_output_space_id": "magenta-oaf-midi88-a0-v1",
    "prediction_schema": "crux.drum-prediction-events/v1",
    "protocol_schema": "crux.transcription-runner/v1",
    "runtime_image_manifest_digest": f"sha256:{'8' * 64}",
    "runtime_lock_sha256": "c" * 64,
    "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
    "upstream_source_commit": "9" * 40,
}
HEURISTIC_DESCRIPTOR = {
    "adapter_source_manifest_sha256": "7" * 64,
    "architecture_id": "librosa-onset-centroid-zcr-v1",
    "backend_id": "heuristic-onset-v1",
    "descriptor_schema": "crux.heuristic-backend-descriptor/v1",
    "model_id": "crux-heuristic-onset-nonmodel-v1",
    "native_metadata_schema_id": "crux-empty-native-metadata-v1",
    "native_output_space_id": "crux-heuristic-midi7-v1",
    "parameter_lock_sha256": "6" * 64,
    "prediction_schema": "crux.drum-prediction-events/v1",
}


def artifact_reference(path: str, digest: str = "a" * 64) -> dict[str, object]:
    return {"path": path, "sha256": digest}


def make_verification_payload(**changes: object) -> dict[str, object]:
    descriptor = dict(OAF_DESCRIPTOR)
    payload: dict[str, object] = {
        "schema": "crux.backend-verification-report/v1",
        "report_type": "verification",
        "run_id": str(FIXED_UUID),
        "started_at": "2026-07-27T01:02:01.000000Z",
        "finished_at": "2026-07-27T01:02:03.456789Z",
        "status": "verified",
        "exit_code": 0,
        "descriptor": descriptor,
        "descriptor_sha256": sha256_hex(canonical_json_bytes(descriptor)),
        "backend_lock_sha256": "b" * 64,
        "runtime_lock_sha256": "c" * 64,
        "parameter_lock_sha256": None,
        "seal_evidence_sha256": "d" * 64,
        "execution_attestation": artifact_reference(
            "artifacts/benchmark/backends/oaf/attestation.json", "e" * 64
        ),
        "tensor_coverage": {
            "status": "passed",
            "required_count": 78,
            "restored_count": 78,
            "non_inference_count": 52,
            "required_inventory_sha256": "f" * 64,
            "non_inference_inventory_sha256": "1" * 64,
            "report_path": "artifacts/benchmark/backends/oaf/tensor.json",
            "report_sha256": "2" * 64,
        },
        "smoke": {
            "status": "passed",
            "audio_sha256": "3" * 64,
            "oracle_sha256": "4" * 64,
            "prediction_path": "artifacts/benchmark/backends/oaf/smoke.jsonl",
            "prediction_sha256": "5" * 64,
        },
        "artifacts": [
            {
                "role": "tensor_coverage",
                "path": "artifacts/benchmark/backends/oaf/tensor.json",
                "sha256": "2" * 64,
            }
        ],
        "errors": [],
    }
    payload.update(changes)
    return payload


def make_execution_item(
    source_audio_id: str,
    *,
    status: str = "complete",
    prediction: object = None,
    midi: object = None,
    errors: list[object] | None = None,
) -> dict[str, object]:
    source_sha256 = sha256_hex(source_audio_id.encode())
    input_view_id = f"{source_audio_id}-view"
    input_sha256 = sha256_hex(input_view_id.encode())
    if prediction is None and status == "complete":
        prediction = artifact_reference(f"predictions/{source_audio_id}.jsonl")
    return {
        "item_id": derive_item_id(
            source_audio_id=source_audio_id,
            source_audio_sha256=source_sha256,
            input_view_id=input_view_id,
            input_audio_sha256=input_sha256,
        ),
        "source_audio_id": source_audio_id,
        "source_audio_sha256": source_sha256,
        "input_view_id": input_view_id,
        "input_audio_sha256": input_sha256,
        "status": status,
        "prediction": prediction,
        "midi": midi,
        "errors": [] if errors is None else errors,
    }


def make_execution_payload(**changes: object) -> dict[str, object]:
    descriptor = dict(OAF_DESCRIPTOR)
    payload: dict[str, object] = {
        "schema": "crux.backend-execution-report/v1",
        "report_type": "execution",
        "run_id": str(FIXED_UUID),
        "started_at": "2026-07-27T01:02:01.000000Z",
        "finished_at": "2026-07-27T01:02:03.456789Z",
        "status": "complete",
        "exit_code": 0,
        "descriptor": descriptor,
        "descriptor_sha256": sha256_hex(canonical_json_bytes(descriptor)),
        "backend_lock_sha256": "b" * 64,
        "runtime_lock_sha256": "c" * 64,
        "parameter_lock_sha256": None,
        "seal_evidence_sha256": "d" * 64,
        "execution_attestation": artifact_reference(
            "artifacts/benchmark/backends/oaf/attestation.json", "e" * 64
        ),
        "verification_report": artifact_reference(
            "artifacts/benchmark/backends/oaf/verification.json", "f" * 64
        ),
        "items": [make_execution_item("song-one")],
        "errors": [],
    }
    payload.update(changes)
    return payload


def make_legacy_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "crux.legacy-score-report/v1",
        "report_type": "legacy_score",
        "run_id": str(FIXED_UUID),
        "started_at": "2026-07-27T01:02:01.000000Z",
        "finished_at": "2026-07-27T01:02:03.456789Z",
        "workflow_mode": "legacy_compatibility",
        "backend_id": "legacy-tf2-h5-v0",
        "backend_validation_status": "unvalidated",
        "status": "complete",
        "exit_code": 0,
        "score_report": artifact_reference("artifacts/benchmark/score.json"),
        "errors": [],
    }
    payload.update(changes)
    return payload


def test_derive_item_id_uses_exact_identity_payload() -> None:
    item_id = derive_item_id(
        source_audio_id="source",
        source_audio_sha256="a" * 64,
        input_view_id="view",
        input_audio_sha256="b" * 64,
    )

    assert item_id == "sha256:1fe66e155d3878095f4540c496c262de08ae3e4e7ca2702b8d29e81117d07683"


def test_derive_item_id_excludes_backend_run_and_output_identity() -> None:
    first = derive_item_id(
        source_audio_id="source",
        source_audio_sha256="a" * 64,
        input_view_id="view",
        input_audio_sha256="b" * 64,
    )
    second = derive_item_id(
        source_audio_id="source",
        source_audio_sha256="a" * 64,
        input_view_id="view",
        input_audio_sha256="b" * 64,
    )

    assert first == second
    assert first.startswith("sha256:")


def test_report_records_require_exact_top_level_keys_and_schemas() -> None:
    verification = VerificationReport(make_verification_payload())
    execution = ExecutionReport(make_execution_payload())
    legacy = LegacyScoreReport(make_legacy_payload())

    assert set(verification.payload) == VERIFICATION_KEYS
    assert set(execution.payload) == EXECUTION_KEYS
    assert set(legacy.payload) == LEGACY_KEYS
    assert verification.payload["schema"] == "crux.backend-verification-report/v1"
    assert execution.payload["schema"] == "crux.backend-execution-report/v1"
    assert legacy.payload["schema"] == "crux.legacy-score-report/v1"


@pytest.mark.parametrize(
    ("record_type", "payload"),
    [
        (VerificationReport, make_verification_payload(unknown=None)),
        (ExecutionReport, make_execution_payload(unknown=None)),
        (LegacyScoreReport, make_legacy_payload(unknown=None)),
    ],
)
def test_report_records_reject_unknown_top_level_keys(
    record_type: type,
    payload: dict[str, object],
) -> None:
    with pytest.raises(ReportValidationError, match="exact key"):
        record_type(payload)


@pytest.mark.parametrize(
    ("record_type", "payload", "bad_status"),
    [
        (VerificationReport, make_verification_payload(), "complete"),
        (ExecutionReport, make_execution_payload(), "verified"),
        (LegacyScoreReport, make_legacy_payload(), "partial"),
    ],
)
def test_report_records_reject_status_outside_exact_enum(
    record_type: type,
    payload: dict[str, object],
    bad_status: str,
) -> None:
    payload["status"] = bad_status

    with pytest.raises(ReportValidationError, match="status"):
        record_type(payload)


@pytest.mark.parametrize(
    ("record_type", "payload", "status", "expected_exit"),
    [
        (VerificationReport, make_verification_payload(), "verified", 0),
        (VerificationReport, make_verification_payload(), "failed", 2),
        (VerificationReport, make_verification_payload(), "environment_unsupported", 1),
        (ExecutionReport, make_execution_payload(), "complete", 0),
        (ExecutionReport, make_execution_payload(), "partial", 1),
        (ExecutionReport, make_execution_payload(), "failed", 2),
        (ExecutionReport, make_execution_payload(), "environment_unsupported", 1),
        (LegacyScoreReport, make_legacy_payload(), "complete", 0),
        (LegacyScoreReport, make_legacy_payload(), "failed", 1),
    ],
)
def test_report_statuses_require_exact_exit_codes(
    record_type: type,
    payload: dict[str, object],
    status: str,
    expected_exit: int,
) -> None:
    payload["status"] = status
    payload["exit_code"] = 99 if expected_exit != 99 else 98

    with pytest.raises(ReportValidationError, match="exit_code"):
        record_type(payload)


def test_errors_and_verification_artifacts_are_sorted_canonically() -> None:
    report = VerificationReport(
        make_verification_payload(
            artifacts=[
                {"role": "z", "path": "b.json", "sha256": "b" * 64},
                {"role": "a", "path": "z.json", "sha256": "c" * 64},
                {"role": "a", "path": "a.json", "sha256": "d" * 64},
            ],
            errors=[
                {"code": "z_error", "message": "first"},
                {"code": "a_error", "message": "z message"},
                {"code": "a_error", "message": "a message"},
            ],
        )
    )

    assert report.payload["artifacts"] == [
        {"role": "a", "path": "a.json", "sha256": "d" * 64},
        {"role": "a", "path": "z.json", "sha256": "c" * 64},
        {"role": "z", "path": "b.json", "sha256": "b" * 64},
    ]
    assert report.payload["errors"] == [
        {"code": "a_error", "message": "a message"},
        {"code": "a_error", "message": "z message"},
        {"code": "z_error", "message": "first"},
    ]


def test_failure_before_identity_establishment_allows_nullable_fields() -> None:
    report = ExecutionReport(
        make_execution_payload(
            status="failed",
            exit_code=2,
            descriptor=None,
            descriptor_sha256=None,
            backend_lock_sha256=None,
            runtime_lock_sha256=None,
            parameter_lock_sha256=None,
            seal_evidence_sha256=None,
            execution_attestation=None,
            verification_report=None,
            items=[],
            errors=[{"code": "backend_unavailable", "message": "Backend is unavailable."}],
        )
    )

    assert report.payload["descriptor"] is None
    assert report.payload["execution_attestation"] is None


@pytest.mark.parametrize(
    "field",
    [
        "descriptor",
        "descriptor_sha256",
        "backend_lock_sha256",
        "runtime_lock_sha256",
        "seal_evidence_sha256",
        "execution_attestation",
    ],
)
def test_verified_oaf_requires_identity_and_attestation_fields(field: str) -> None:
    payload = make_verification_payload()
    payload[field] = None

    with pytest.raises(ReportValidationError, match=field):
        VerificationReport(payload)


def test_verified_oaf_requires_passed_tensor_coverage_and_smoke() -> None:
    payload = make_verification_payload()
    tensor_coverage = dict(payload["tensor_coverage"])  # type: ignore[arg-type]
    tensor_coverage["status"] = "not_run"
    payload["tensor_coverage"] = tensor_coverage

    with pytest.raises(ReportValidationError, match="tensor_coverage"):
        VerificationReport(payload)


def test_successful_heuristic_uses_parameter_identity_and_non_applicable_checks() -> None:
    descriptor = dict(HEURISTIC_DESCRIPTOR)
    report = VerificationReport(
        make_verification_payload(
            descriptor=descriptor,
            descriptor_sha256=sha256_hex(canonical_json_bytes(descriptor)),
            backend_lock_sha256=None,
            runtime_lock_sha256=None,
            parameter_lock_sha256="6" * 64,
            seal_evidence_sha256=None,
            tensor_coverage={
                "status": "not_applicable",
                "required_count": 0,
                "restored_count": 0,
                "non_inference_count": 0,
                "required_inventory_sha256": None,
                "non_inference_inventory_sha256": None,
                "report_path": None,
                "report_sha256": None,
            },
            smoke={
                "status": "not_applicable",
                "audio_sha256": None,
                "oracle_sha256": None,
                "prediction_path": None,
                "prediction_sha256": None,
            },
            artifacts=[],
        )
    )

    assert report.payload["parameter_lock_sha256"] == "6" * 64
    assert report.payload["backend_lock_sha256"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda descriptor: descriptor.update({"unexpected": "value"}), "exact key"),
        (lambda descriptor: descriptor.pop("model_id"), "exact key"),
        (
            lambda descriptor: descriptor.update({"descriptor_schema": "crux.unknown/v1"}),
            "descriptor_schema",
        ),
        (lambda descriptor: descriptor.update({"backend_id": "unknown-backend-v1"}), "backend_id"),
    ],
)
def test_report_descriptor_requires_exact_known_backend_shape(
    mutation: object,
    message: str,
) -> None:
    descriptor = dict(OAF_DESCRIPTOR)
    mutation(descriptor)  # type: ignore[operator]

    with pytest.raises(ReportValidationError, match=message):
        VerificationReport(
            make_verification_payload(
                descriptor=descriptor,
                descriptor_sha256=sha256_hex(canonical_json_bytes(descriptor)),
            )
        )


@pytest.mark.parametrize(
    ("descriptor_field", "report_field"),
    [
        ("backend_lock_sha256", "backend_lock_sha256"),
        ("runtime_lock_sha256", "runtime_lock_sha256"),
    ],
)
def test_oaf_descriptor_lock_identity_must_match_report(
    descriptor_field: str,
    report_field: str,
) -> None:
    descriptor = dict(OAF_DESCRIPTOR)
    descriptor[descriptor_field] = "a" * 64
    payload = make_verification_payload(
        descriptor=descriptor,
        descriptor_sha256=sha256_hex(canonical_json_bytes(descriptor)),
    )
    payload[report_field] = "b" * 64 if report_field == "backend_lock_sha256" else "c" * 64

    with pytest.raises(ReportValidationError, match=report_field):
        VerificationReport(payload)


def test_oaf_descriptor_requires_exact_runner_protocol_schema() -> None:
    descriptor = dict(OAF_DESCRIPTOR)
    descriptor["protocol_schema"] = "crux.transcription-runner-wrong/v1"

    with pytest.raises(
        ReportValidationError,
        match="descriptor protocol_schema does not match frozen identity",
    ):
        VerificationReport(
            make_verification_payload(
                descriptor=descriptor,
                descriptor_sha256=sha256_hex(canonical_json_bytes(descriptor)),
            )
        )


def test_heuristic_descriptor_parameter_lock_must_match_report() -> None:
    descriptor = dict(HEURISTIC_DESCRIPTOR)
    descriptor["parameter_lock_sha256"] = "a" * 64

    with pytest.raises(ReportValidationError, match="parameter_lock_sha256"):
        ExecutionReport(
            make_execution_payload(
                descriptor=descriptor,
                descriptor_sha256=sha256_hex(canonical_json_bytes(descriptor)),
                backend_lock_sha256=None,
                runtime_lock_sha256=None,
                parameter_lock_sha256="6" * 64,
                seal_evidence_sha256=None,
            )
        )


def test_execution_items_remain_in_request_order() -> None:
    report = ExecutionReport(
        make_execution_payload(
            items=[
                make_execution_item("requested-first"),
                make_execution_item("requested-second"),
            ]
        )
    )

    items = report.payload["items"]
    assert isinstance(items, list)
    assert [item["source_audio_id"] for item in items] == [  # type: ignore[index]
        "requested-first",
        "requested-second",
    ]


def test_complete_item_allows_null_midi_when_midi_was_not_requested() -> None:
    report = ExecutionReport(
        make_execution_payload(
            items=[make_execution_item("prediction-only", midi=None)],
        )
    )

    item = report.payload["items"][0]  # type: ignore[index]
    assert item["status"] == "complete"
    assert item["prediction"] is not None
    assert item["midi"] is None


def test_midi_failure_can_retain_prediction_as_incomplete_item() -> None:
    item = make_execution_item(
        "song",
        status="incomplete",
        prediction=artifact_reference("predictions/song.jsonl"),
        midi=None,
        errors=[{"code": "midi_derivation_failed", "message": "MIDI was not produced."}],
    )

    report = ExecutionReport(make_execution_payload(status="partial", exit_code=1, items=[item]))

    stored = report.payload["items"]
    assert isinstance(stored, list)
    assert stored[0]["prediction"] == artifact_reference("predictions/song.jsonl")
    assert stored[0]["midi"] is None
    assert stored[0]["status"] == "incomplete"


@pytest.mark.parametrize(
    "field",
    [
        "backend_lock_sha256",
        "runtime_lock_sha256",
        "seal_evidence_sha256",
        "execution_attestation",
        "verification_report",
    ],
)
def test_complete_oaf_execution_requires_established_backend_relationships(
    field: str,
) -> None:
    payload = make_execution_payload()
    payload[field] = None

    with pytest.raises(ReportValidationError, match=field):
        ExecutionReport(payload)


def test_complete_heuristic_execution_requires_parameter_lock_only() -> None:
    descriptor = dict(HEURISTIC_DESCRIPTOR)
    report = ExecutionReport(
        make_execution_payload(
            descriptor=descriptor,
            descriptor_sha256=sha256_hex(canonical_json_bytes(descriptor)),
            backend_lock_sha256=None,
            runtime_lock_sha256=None,
            parameter_lock_sha256="6" * 64,
            seal_evidence_sha256=None,
        )
    )

    assert report.payload["parameter_lock_sha256"] == "6" * 64
    assert report.payload["runtime_lock_sha256"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "12345678-1234-1678-9234-567812345678"),
        ("run_id", "12345678-1234-4678-9234-56781234567A"),
        ("started_at", "2026-07-27T01:02:01Z"),
        ("finished_at", "2026-07-27T01:02:03.456789+00:00"),
    ],
)
def test_reports_require_lowercase_uuidv4_and_microsecond_utc_timestamps(
    field: str,
    value: str,
) -> None:
    payload = make_execution_payload()
    payload[field] = value

    with pytest.raises(ReportValidationError, match=field):
        ExecutionReport(payload)


def test_report_publication_updates_latest_after_immutable_report(tmp_path: Path) -> None:
    report = ExecutionReport(make_execution_payload())

    published = publish_operational_report(
        tmp_path,
        backend_id=OAF_BACKEND_ID,
        report=report,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    latest = tmp_path / OAF_BACKEND_ID / "latest-execution.json"
    assert published.path.exists()
    assert latest.read_bytes() == published.path.read_bytes()
    assert published.path == (
        tmp_path
        / OAF_BACKEND_ID
        / "reports"
        / "20260727T010203456789Z-12345678-1234-4678-9234-567812345678.json"
    )
    assert published.role == "execution_report"


def test_report_publication_requires_payload_backend_namespace_match(tmp_path: Path) -> None:
    report = ExecutionReport(make_execution_payload())

    with pytest.raises(OperationalReportPublicationError):
        publish_operational_report(
            tmp_path,
            backend_id="heuristic-onset-v1",
            report=report,
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert not (tmp_path / "heuristic-onset-v1").exists()


def test_legacy_report_publication_uses_its_declared_namespace(tmp_path: Path) -> None:
    published = publish_operational_report(
        tmp_path,
        backend_id="legacy-tf2-h5-v0",
        report=LegacyScoreReport(make_legacy_payload()),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert published.path.parent.parent.name == "legacy-tf2-h5-v0"


def test_report_uses_detached_construction_snapshot_for_publication(tmp_path: Path) -> None:
    payload = make_execution_payload()
    report = ExecutionReport(payload)
    original_items = payload["items"]
    assert isinstance(original_items, list)
    original_items[0]["source_audio_id"] = "caller-tampered"  # type: ignore[index]
    exposed_items = report.payload["items"]
    assert isinstance(exposed_items, list)
    exposed_items[0]["source_audio_id"] = "exposed-tampered"  # type: ignore[index]
    exposed_items.append(make_execution_item("injected"))

    published = publish_operational_report(
        tmp_path,
        backend_id=OAF_BACKEND_ID,
        report=report,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )
    persisted = strict_json_loads(published.path.read_bytes()[:-1], require_canonical=True)

    assert isinstance(persisted, dict)
    assert [item["source_audio_id"] for item in persisted["items"]] == ["song-one"]


def test_report_run_id_and_publication_run_id_must_match(tmp_path: Path) -> None:
    report = ExecutionReport(make_execution_payload())

    with pytest.raises(OperationalReportPublicationError):
        publish_operational_report(
            tmp_path,
            backend_id=OAF_BACKEND_ID,
            report=report,
            now=FIXED_UTC,
            run_id=UUID("87654321-4321-4765-8765-432187654321"),
        )

    assert not (tmp_path / OAF_BACKEND_ID).exists()


def test_same_timestamp_different_run_ids_have_distinct_filenames(tmp_path: Path) -> None:
    first = publish_operational_report(
        tmp_path,
        backend_id=OAF_BACKEND_ID,
        report=ExecutionReport(make_execution_payload()),
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )
    second_uuid = UUID("87654321-4321-4765-8765-432187654321")
    second = publish_operational_report(
        tmp_path,
        backend_id=OAF_BACKEND_ID,
        report=ExecutionReport(make_execution_payload(run_id=str(second_uuid))),
        now=FIXED_UTC,
        run_id=second_uuid,
    )

    assert first.path != second.path
    assert first.path.exists()
    assert second.path.exists()


def test_existing_immutable_report_content_mismatch_is_rejected(tmp_path: Path) -> None:
    report = ExecutionReport(make_execution_payload())
    report_path = (
        tmp_path
        / OAF_BACKEND_ID
        / "reports"
        / "20260727T010203456789Z-12345678-1234-4678-9234-567812345678.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(b"different immutable bytes\n")

    with pytest.raises(OperationalReportPublicationError):
        publish_operational_report(
            tmp_path,
            backend_id=OAF_BACKEND_ID,
            report=report,
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert report_path.read_bytes() == b"different immutable bytes\n"


@pytest.mark.parametrize("latest_name", ["latest-execution.json", "latest-verification.json"])
def test_report_publication_rejects_symlink_latest_destination(
    tmp_path: Path,
    latest_name: str,
) -> None:
    backend_root = tmp_path / OAF_BACKEND_ID
    backend_root.mkdir()
    target = tmp_path / "outside.json"
    target.write_bytes(b"outside")
    (backend_root / latest_name).symlink_to(target)
    report = (
        ExecutionReport(make_execution_payload())
        if latest_name == "latest-execution.json"
        else VerificationReport(make_verification_payload())
    )

    with pytest.raises(OperationalReportPublicationError):
        publish_operational_report(
            tmp_path,
            backend_id=OAF_BACKEND_ID,
            report=report,
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert target.read_bytes() == b"outside"


def test_report_latest_replace_failure_rolls_back_previous_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_root = tmp_path / OAF_BACKEND_ID
    backend_root.mkdir()
    latest = backend_root / "latest-execution.json"
    latest.write_bytes(b"prior valid latest\n")

    def failed_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(backend_publication.os, "replace", failed_replace)

    with pytest.raises(OperationalReportPublicationError):
        publish_operational_report(
            tmp_path,
            backend_id=OAF_BACKEND_ID,
            report=ExecutionReport(make_execution_payload()),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert latest.read_bytes() == b"prior valid latest\n"


def test_latest_snapshot_failure_leaves_existing_inode_and_bytes_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_root = tmp_path / OAF_BACKEND_ID
    backend_root.mkdir()
    latest = backend_root / "latest-execution.json"
    latest.write_bytes(b"prior valid latest\n")
    prior_stat = latest.stat()

    def fail_snapshot(_path: Path) -> bytes | None:
        raise OSError("snapshot failed")

    monkeypatch.setattr(backend_reports, "_read_optional_regular_file", fail_snapshot)

    with pytest.raises(OperationalReportPublicationError):
        publish_operational_report(
            tmp_path,
            backend_id=OAF_BACKEND_ID,
            report=ExecutionReport(make_execution_payload()),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert latest.read_bytes() == b"prior valid latest\n"
    assert latest.stat().st_ino == prior_stat.st_ino


def test_failed_older_latest_publish_cannot_roll_back_newer_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pylint: disable=too-many-locals
    backend_root = tmp_path / OAF_BACKEND_ID
    backend_root.mkdir()
    latest = backend_root / "latest-execution.json"
    latest.write_bytes(b"original latest\n")
    newer_run_id = UUID("87654321-4321-4765-8765-432187654321")
    older_report = ExecutionReport(make_execution_payload())
    newer_report = ExecutionReport(make_execution_payload(run_id=str(newer_run_id)))
    real_replace = backend_reports.atomic_replace_bytes
    real_fsync_directory = backend_reports.fsync_directory
    newer_started = threading.Event()
    newer_replaced = threading.Event()
    older_at_fsync = threading.Event()
    older_failed_once = False
    failures: list[OperationalReportPublicationError] = []
    newer_publication: list[object] = []

    def observe_replace(path: Path, content: bytes) -> None:
        real_replace(path, content)
        if threading.current_thread().name == "newer" and path == latest:
            newer_replaced.set()

    def fail_older_after_replace(path: Path) -> None:
        nonlocal older_failed_once
        if (
            threading.current_thread().name == "older"
            and path == backend_root
            and not older_failed_once
        ):
            older_failed_once = True
            older_at_fsync.set()
            assert newer_started.wait(timeout=2)
            newer_replaced.wait(timeout=0.25)
            raise OSError("older durability failure")
        real_fsync_directory(path)

    def publish_older() -> None:
        try:
            publish_operational_report(
                tmp_path,
                backend_id=OAF_BACKEND_ID,
                report=older_report,
                now=FIXED_UTC,
                run_id=FIXED_UUID,
            )
        except OperationalReportPublicationError as error:
            failures.append(error)

    def publish_newer() -> None:
        newer_started.set()
        newer_publication.append(
            publish_operational_report(
                tmp_path,
                backend_id=OAF_BACKEND_ID,
                report=newer_report,
                now=FIXED_UTC,
                run_id=newer_run_id,
            )
        )

    monkeypatch.setattr(backend_reports, "atomic_replace_bytes", observe_replace)
    monkeypatch.setattr(backend_reports, "fsync_directory", fail_older_after_replace)
    older = threading.Thread(target=publish_older, name="older")
    newer = threading.Thread(target=publish_newer, name="newer")
    older.start()
    assert older_at_fsync.wait(timeout=2)
    newer.start()
    older.join(timeout=5)
    newer.join(timeout=5)

    assert not older.is_alive()
    assert not newer.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], OperationalReportPublicationError)
    assert len(newer_publication) == 1
    expected = (
        backend_root
        / "reports"
        / "20260727T010203456789Z-87654321-4321-4765-8765-432187654321.json"
    )
    assert latest.read_bytes() == expected.read_bytes()


def test_report_latest_directory_fsync_failure_rolls_back_previous_latest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_root = tmp_path / OAF_BACKEND_ID
    backend_root.mkdir()
    latest = backend_root / "latest-execution.json"
    latest.write_bytes(b"prior valid latest\n")
    real_fsync_directory = backend_reports.fsync_directory
    failed_once = False

    def fail_after_latest_replace(path: Path) -> None:
        nonlocal failed_once
        if path == backend_root and not failed_once:
            failed_once = True
            raise OSError("directory fsync failed")
        real_fsync_directory(path)

    monkeypatch.setattr(backend_reports, "fsync_directory", fail_after_latest_replace)

    with pytest.raises(OperationalReportPublicationError):
        publish_operational_report(
            tmp_path,
            backend_id=OAF_BACKEND_ID,
            report=ExecutionReport(make_execution_payload()),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert latest.read_bytes() == b"prior valid latest\n"


def test_report_failure_preserves_previously_published_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = tmp_path / "predictions" / "result.jsonl"
    prediction.parent.mkdir()
    prediction.write_bytes(b"prior valid prediction\n")

    def failed_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(backend_publication.os, "replace", failed_replace)

    with pytest.raises(OperationalReportPublicationError):
        publish_operational_report(
            tmp_path,
            backend_id=OAF_BACKEND_ID,
            report=ExecutionReport(make_execution_payload()),
            now=FIXED_UTC,
            run_id=FIXED_UUID,
        )

    assert prediction.read_bytes() == b"prior valid prediction\n"


def test_legacy_canonical_mapping_required_has_null_score_and_exit_one() -> None:
    report = LegacyScoreReport(
        make_legacy_payload(
            backend_id="magenta-egmd-tf1-94529798-8hit-v1",
            backend_validation_status="not_checked",
            status="canonical_mapping_required",
            exit_code=1,
            score_report=None,
        )
    )

    assert report.payload["score_report"] is None
    assert report.payload["status"] == "canonical_mapping_required"


def test_report_bytes_are_canonical_with_one_final_newline(tmp_path: Path) -> None:
    report = LegacyScoreReport(make_legacy_payload())

    published = publish_operational_report(
        tmp_path,
        backend_id="legacy-tf2-h5-v0",
        report=report,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )
    content = published.path.read_bytes()

    assert content.endswith(b"\n")
    assert not content.endswith(b"\n\n")
    assert strict_json_loads(content[:-1], require_canonical=True) == dict(report.payload)
