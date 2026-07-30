from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Literal, cast

from src.benchmark.backend_identity import (
    StrictJsonError,
    canonical_json_bytes,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backend_publication import read_regular_file_no_follow
from src.benchmark.backend_registry import OFFICIAL_BACKEND_ID

CHECKPOINT_ACQUISITION_REQUEST_SCHEMA = "crux.oaf-checkpoint-acquisition-request/v1"
CHECKPOINT_ACQUISITION_EVIDENCE_SCHEMA = "crux.oaf-checkpoint-acquisition-evidence/v1"
CHECKPOINT_URL = (
    "https://storage.googleapis.com/magentadata/models/"
    "onsets_frames_transcription/e-gmd_checkpoint.zip"
)
_ARCHIVE = (
    "e-gmd_checkpoint.zip",
    "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0",
    25658703,
)
_MEMBERS = (
    (
        "checkpoint",
        "c4afc7e992f63a290fc9b061bc36582eb08db5f4c10f8b79971982217f039a2b",
        91,
        "pointer",
    ),
    (
        "model.ckpt-569400.data-00000-of-00001",
        "6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5",
        27793012,
        "published_component",
    ),
    (
        "model.ckpt-569400.index",
        "475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a",
        2713,
        "published_component",
    ),
    (
        "model.ckpt-569400.meta",
        "e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422",
        3640417,
        "published_component",
    ),
)
_REQUEST_KEYS = frozenset(
    {
        "schema",
        "backend_id",
        "checkpoint_url",
        "archive",
        "archive_members",
        "published_component_names",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "request_sha256",
        "acquisition_mode",
        "archive",
        "archive_members",
        "published_components",
        "model_artifact_set_sha256",
        "cache_path",
    }
)


class CheckpointAcquisitionError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointIdentity:
    name: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ArchiveMemberIdentity(CheckpointIdentity):
    role: Literal["pointer", "published_component"]


@dataclass(frozen=True)
class CheckpointAcquisitionRequest:
    backend_id: str
    checkpoint_url: str
    archive: CheckpointIdentity
    archive_members: tuple[ArchiveMemberIdentity, ...]
    published_component_names: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class CheckpointAcquisitionEvidence:
    request_sha256: str
    acquisition_mode: Literal["download", "archive", "cache_verify"]
    model_artifact_set_sha256: str
    cache_path: PurePosixPath
    sha256: str


def render_checkpoint_acquisition_evidence(
    request: CheckpointAcquisitionRequest,
    *,
    acquisition_mode: Literal["download", "archive", "cache_verify"],
    model_artifact_set_sha256: str,
    cache_path: PurePosixPath,
) -> tuple[CheckpointAcquisitionEvidence, bytes]:
    """Render the immutable setup evidence after the cache was reverified."""
    if cache_path.is_absolute() or ".." in cache_path.parts or not cache_path.parts:
        raise CheckpointAcquisitionError("checkpoint evidence cache path is invalid")
    require_sha256(model_artifact_set_sha256, "checkpoint evidence artifact set")
    payload = {
        "acquisition_mode": acquisition_mode,
        "archive": _identity_payload(request.archive),
        "archive_members": [
            {**_identity_payload(member), "role": member.role} for member in request.archive_members
        ],
        "cache_path": str(cache_path),
        "model_artifact_set_sha256": model_artifact_set_sha256,
        "published_components": [
            _identity_payload(member)
            for member in request.archive_members
            if member.role == "published_component"
        ],
        "request_sha256": request.sha256,
        "schema": CHECKPOINT_ACQUISITION_EVIDENCE_SCHEMA,
    }
    content = canonical_json_bytes(payload, trailing_newline=True)
    return (
        CheckpointAcquisitionEvidence(
            request_sha256=request.sha256,
            acquisition_mode=acquisition_mode,
            model_artifact_set_sha256=model_artifact_set_sha256,
            cache_path=cache_path,
            sha256=sha256_hex(content),
        ),
        content,
    )


def load_checkpoint_acquisition_request(path: Path) -> CheckpointAcquisitionRequest:
    """Load the fixed pre-seal checkpoint authority without following links."""
    try:
        content = read_regular_file_no_follow(path)
        if not content.endswith(b"\n") or content.endswith(b"\n\n"):
            raise CheckpointAcquisitionError(
                "checkpoint acquisition request must have one final newline"
            )
        value = strict_json_loads(content[:-1], require_canonical=True)
        if not isinstance(value, dict) or set(value) != _REQUEST_KEYS:
            raise CheckpointAcquisitionError("checkpoint acquisition request key set is invalid")
        if value["schema"] != CHECKPOINT_ACQUISITION_REQUEST_SCHEMA:
            raise CheckpointAcquisitionError("checkpoint acquisition request schema is invalid")
        if value["backend_id"] != OFFICIAL_BACKEND_ID or value["checkpoint_url"] != CHECKPOINT_URL:
            raise CheckpointAcquisitionError("checkpoint acquisition request identity is invalid")
        archive = _identity(value["archive"], "archive")
        if (archive.name, archive.sha256, archive.size) != _ARCHIVE:
            raise CheckpointAcquisitionError("checkpoint acquisition archive is invalid")
        members_value = value["archive_members"]
        if not isinstance(members_value, list):
            raise CheckpointAcquisitionError("checkpoint acquisition archive members are invalid")
        members = tuple(_member(member) for member in members_value)
        if (
            tuple((member.name, member.sha256, member.size, member.role) for member in members)
            != _MEMBERS
        ):
            raise CheckpointAcquisitionError("checkpoint acquisition archive members are invalid")
        names_value = value["published_component_names"]
        expected_names = tuple(
            member[0] for member in _MEMBERS if member[3] == "published_component"
        )
        if not isinstance(names_value, list) or tuple(names_value) != expected_names:
            raise CheckpointAcquisitionError(
                "checkpoint acquisition published components are invalid"
            )
        return CheckpointAcquisitionRequest(
            backend_id=OFFICIAL_BACKEND_ID,
            checkpoint_url=CHECKPOINT_URL,
            archive=archive,
            archive_members=members,
            published_component_names=expected_names,
            sha256=sha256_hex(content),
        )
    except (OSError, StrictJsonError, KeyError, TypeError, CheckpointAcquisitionError) as error:
        if isinstance(error, CheckpointAcquisitionError):
            raise
        raise CheckpointAcquisitionError("checkpoint acquisition request is invalid") from None


def load_checkpoint_acquisition_evidence(
    path: Path,
    *,
    request: CheckpointAcquisitionRequest,
) -> CheckpointAcquisitionEvidence:
    """Load evidence only when it exactly reproduces the authenticated request."""
    try:
        content = read_regular_file_no_follow(path)
        if not content.endswith(b"\n") or content.endswith(b"\n\n"):
            raise CheckpointAcquisitionError(
                "checkpoint acquisition evidence must have one final newline"
            )
        value = strict_json_loads(content[:-1], require_canonical=True)
        if not isinstance(value, dict) or set(value) != _EVIDENCE_KEYS:
            raise CheckpointAcquisitionError("checkpoint acquisition evidence key set is invalid")
        if value["schema"] != CHECKPOINT_ACQUISITION_EVIDENCE_SCHEMA:
            raise CheckpointAcquisitionError("checkpoint acquisition evidence schema is invalid")
        mode = value["acquisition_mode"]
        if mode not in {"download", "archive", "cache_verify"}:
            raise CheckpointAcquisitionError("checkpoint acquisition evidence mode is invalid")
        if value["request_sha256"] != request.sha256:
            raise CheckpointAcquisitionError(
                "checkpoint acquisition evidence request hash is invalid"
            )
        if value["archive"] != _identity_payload(request.archive):
            raise CheckpointAcquisitionError("checkpoint acquisition evidence archive is invalid")
        expected_members = [
            {**_identity_payload(member), "role": member.role} for member in request.archive_members
        ]
        if value["archive_members"] != expected_members:
            raise CheckpointAcquisitionError("checkpoint acquisition evidence members are invalid")
        expected_components = [
            _identity_payload(member)
            for member in request.archive_members
            if member.role == "published_component"
        ]
        if value["published_components"] != expected_components:
            raise CheckpointAcquisitionError(
                "checkpoint acquisition evidence components are invalid"
            )
        artifact_hash = value["model_artifact_set_sha256"]
        cache_path = value["cache_path"]
        if not isinstance(artifact_hash, str) or not isinstance(cache_path, str):
            raise CheckpointAcquisitionError("checkpoint acquisition evidence fields are invalid")
        require_sha256(artifact_hash, "checkpoint evidence artifact set")
        parsed_path = PurePosixPath(cache_path)
        if parsed_path.is_absolute() or ".." in parsed_path.parts or not parsed_path.parts:
            raise CheckpointAcquisitionError("checkpoint evidence cache path is invalid")
        return CheckpointAcquisitionEvidence(
            request_sha256=request.sha256,
            acquisition_mode=cast(Literal["download", "archive", "cache_verify"], mode),
            model_artifact_set_sha256=artifact_hash,
            cache_path=parsed_path,
            sha256=sha256_hex(content),
        )
    except (OSError, StrictJsonError, KeyError, TypeError, CheckpointAcquisitionError) as error:
        if isinstance(error, CheckpointAcquisitionError):
            raise
        raise CheckpointAcquisitionError("checkpoint acquisition evidence is invalid") from None


def validate_schema_golden(schema: str, content: bytes) -> None:
    """Run isolated acquisition fixtures through production acquisition loaders."""
    try:
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "golden.json"
            path.write_bytes(content)
            if schema == CHECKPOINT_ACQUISITION_REQUEST_SCHEMA:
                load_checkpoint_acquisition_request(path)
            elif schema == CHECKPOINT_ACQUISITION_EVIDENCE_SCHEMA:
                request_path = Path(directory) / "request.json"
                request_path.write_bytes(
                    canonical_json_bytes(
                        {
                            "archive": {
                                "name": _ARCHIVE[0],
                                "sha256": _ARCHIVE[1],
                                "size": _ARCHIVE[2],
                            },
                            "archive_members": [
                                {"name": name, "role": role, "sha256": digest, "size": size}
                                for name, digest, size, role in _MEMBERS
                            ],
                            "backend_id": OFFICIAL_BACKEND_ID,
                            "checkpoint_url": CHECKPOINT_URL,
                            "published_component_names": [
                                name
                                for name, _digest, _size, role in _MEMBERS
                                if role == "published_component"
                            ],
                            "schema": CHECKPOINT_ACQUISITION_REQUEST_SCHEMA,
                        },
                        trailing_newline=True,
                    )
                )
                load_checkpoint_acquisition_evidence(
                    path,
                    request=load_checkpoint_acquisition_request(request_path),
                )
            else:
                raise CheckpointAcquisitionError(
                    "checkpoint acquisition schema golden is unsupported"
                )
    except (OSError, CheckpointAcquisitionError) as error:
        raise ValueError(str(error)) from None


def _identity(value: object, field: str) -> CheckpointIdentity:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "size"}:
        raise CheckpointAcquisitionError(f"checkpoint {field} identity is invalid")
    name = value["name"]
    digest = value["sha256"]
    size = value["size"]
    if not isinstance(name, str) or not _safe_name(name):
        raise CheckpointAcquisitionError(f"checkpoint {field} name is invalid")
    if (
        not isinstance(digest, str)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise CheckpointAcquisitionError(f"checkpoint {field} identity is invalid")
    require_sha256(digest, f"checkpoint {field} sha256")
    return CheckpointIdentity(name=name, sha256=digest, size=size)


def _identity_payload(identity: CheckpointIdentity) -> dict[str, object]:
    return {"name": identity.name, "sha256": identity.sha256, "size": identity.size}


def _member(value: object) -> ArchiveMemberIdentity:
    if not isinstance(value, dict) or set(value) != {"name", "sha256", "size", "role"}:
        raise CheckpointAcquisitionError("checkpoint archive member identity is invalid")
    identity = _identity({key: value[key] for key in ("name", "sha256", "size")}, "archive member")
    role = value["role"]
    if role not in {"pointer", "published_component"}:
        raise CheckpointAcquisitionError("checkpoint archive member role is invalid")
    return ArchiveMemberIdentity(
        name=identity.name,
        sha256=identity.sha256,
        size=identity.size,
        role=cast(Literal["pointer", "published_component"], role),
    )


def _safe_name(name: str) -> bool:
    return (
        bool(name)
        and name not in {".", ".."}
        and all(part not in name for part in ("/", "\\", ":", "\x00"))
    )
