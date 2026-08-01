from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import secrets
import stat
import struct
import urllib.request
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal, Protocol, cast
from urllib.parse import urlsplit

from src.benchmark.backend_identity import canonical_json_bytes, sha256_hex
from src.benchmark.backend_lock import (
    BackendLockError,
    LoadedBackendLock,
    revalidate_loaded_backend_lock,
)
from src.benchmark.backend_publication import (
    ArtifactAlreadyPublishedError,
    ArtifactPublicationError,
    DirectoryPublicationError,
    publish_immutable_bytes,
    rename_directory_no_replace,
)
from src.benchmark.backends import PublishedArtifact
from src.benchmark.checkpoint_acquisition import (
    CheckpointAcquisitionError,
    CheckpointAcquisitionRequest,
    load_checkpoint_acquisition_request,
    render_checkpoint_acquisition_evidence,
)

# The ZIP32 validation code is intentionally colocated with the publication
# transaction so the archive trust boundary stays reviewable in one module.
# pylint: disable=too-many-lines

_OFFICIAL_CHECKPOINT_URL = (
    "https://storage.googleapis.com/magentadata/models/"
    "onsets_frames_transcription/e-gmd_checkpoint.zip"
)
_READ_CHUNK_BYTES = 1024 * 1024
_SUPPORTED_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_EOCD = struct.Struct("<4s4H2LH")
_CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_EOCD_SIGNATURE = b"PK\x05\x06"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_LOCAL_SIGNATURE = b"PK\x03\x04"
_DATA_DESCRIPTOR_SIGNATURE = b"PK\x07\x08"
_ZIP64_EXTRA_ID = 0x0001
_ZIP32_SENTINEL = 0xFFFFFFFF
_ALLOWED_GENERAL_FLAGS = 0x080E

PrepareStatus = Literal["ready", "acquisition_failed", "integrity_failed"]
PrepareExitCode = Literal[0, 1, 2]


@dataclass(frozen=True)
class PrepareBackendRequest:
    backend_id: str
    cache_root: Path
    archive_path: Path | None
    download: bool
    acquisition_request_path: Path | None = None
    evidence_output_path: Path | None = None
    backend_lock_path: Path | None = None


@dataclass(frozen=True)
class PrepareBackendOutcome:
    status: PrepareStatus
    exit_code: PrepareExitCode
    model_cache_path: Path | None
    evidence_artifact: PublishedArtifact | None = None


@dataclass(frozen=True)
class _Component:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class _LockContract:
    archive_size: int
    archive_sha256: str
    checkpoint_url: str
    components: tuple[_Component, ...]
    model_artifact_set_sha256: str
    archive_members: tuple[_Component, ...] = ()
    pointer: _Component | None = None


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class _ZipMember:
    name: str
    raw_name: bytes
    flags: int
    compression: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int
    data_offset: int
    region_end: int


class _PrepareError(RuntimeError):
    status: PrepareStatus
    exit_code: PrepareExitCode


class _AcquisitionError(_PrepareError):
    status = "acquisition_failed"
    exit_code = 1


class _IntegrityError(_PrepareError):
    status = "integrity_failed"
    exit_code = 2


class _RedirectRejected(_IntegrityError):
    pass


class _Digest(Protocol):
    def update(self, content: bytes) -> None:
        """Add bytes to the digest."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    # urllib fixes this callback signature.
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> urllib.request.Request | None:
        del req, fp, code, msg, headers, newurl
        raise _RedirectRejected("checkpoint redirect rejected")


def prepare_oaf_backend(
    request: PrepareBackendRequest,
    *,
    backend_lock: LoadedBackendLock | None = None,
) -> PrepareBackendOutcome:
    """Acquire or verify the immutable released OaF checkpoint component set."""
    if backend_lock is None:
        return _prepare_preseal_oaf_backend(request)
    try:
        validated_lock = revalidate_loaded_backend_lock(backend_lock)
    except BackendLockError:
        return _failure(_IntegrityError("backend lock failed strict revalidation"))
    contract = _contract_from_validated_lock(request, validated_lock)
    if isinstance(contract, _PrepareError):
        return _failure(contract)
    if request.backend_lock_path is not None:
        try:
            acquisition_path = request.acquisition_request_path or (
                validated_lock.path.parent
                / f"{request.backend_id}.checkpoint-acquisition-request.json"
            )
            acquisition_request = load_checkpoint_acquisition_request(acquisition_path)
            acquisition_contract = _contract_from_acquisition_request(request, acquisition_request)
            if _contract_identity(acquisition_contract) != _contract_identity(contract):
                return _failure(
                    _IntegrityError("checkpoint request contradicts the final backend lock")
                )
        except CheckpointAcquisitionError:
            return _failure(_IntegrityError("checkpoint acquisition request is invalid"))
    return _prepare_oaf_contract(request, contract)


def _prepare_preseal_oaf_backend(request: PrepareBackendRequest) -> PrepareBackendOutcome:
    if request.acquisition_request_path is None or request.evidence_output_path is None:
        return _failure(_IntegrityError("pre-seal acquisition authority is incomplete"))
    try:
        acquisition_request = load_checkpoint_acquisition_request(request.acquisition_request_path)
        contract = _contract_from_acquisition_request(request, acquisition_request)
    except CheckpointAcquisitionError:
        return _failure(_IntegrityError("checkpoint acquisition request is invalid"))
    outcome = _prepare_oaf_contract(request, contract)
    if outcome.status != "ready" or outcome.model_cache_path is None:
        return outcome
    try:
        cache_path = PurePosixPath(
            outcome.model_cache_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        )
        evidence, content = render_checkpoint_acquisition_evidence(
            acquisition_request,
            acquisition_mode=(
                "download"
                if request.download
                else "archive"
                if request.archive_path is not None
                else "cache_verify"
            ),
            model_artifact_set_sha256=contract.model_artifact_set_sha256,
            cache_path=cache_path,
        )
        published = publish_immutable_bytes(
            request.evidence_output_path,
            content,
            evidence.sha256,
            role="checkpoint_acquisition_evidence",
        )
    except (ArtifactPublicationError, CheckpointAcquisitionError, ValueError):
        return _failure(_IntegrityError("checkpoint acquisition evidence could not be published"))
    return replace(outcome, evidence_artifact=published)


def _prepare_oaf_contract(
    request: PrepareBackendRequest,
    contract: _LockContract,
) -> PrepareBackendOutcome:
    if request.download and request.archive_path is not None:
        return _failure(_IntegrityError("acquisition modes are mutually exclusive"))

    verify_only = not request.download and request.archive_path is None
    try:
        cache_fd = _open_directory_chain(request.cache_root, create=not verify_only)
    except _PrepareError as error:
        return _failure(error)

    try:
        try:
            sha256_fd = _open_child_directory(
                cache_fd,
                "sha256",
                create=not verify_only,
            )
        except _PrepareError as error:
            return _failure(error)
        try:
            if verify_only:
                return _verify_existing_cache(request, contract, sha256_fd)
            return _prepare_under_lock(request, contract, sha256_fd)
        finally:
            os.close(sha256_fd)
    finally:
        os.close(cache_fd)


def _prepare_under_lock(
    request: PrepareBackendRequest,
    contract: _LockContract,
    sha256_fd: int,
) -> PrepareBackendOutcome:
    try:
        with _publication_lock(sha256_fd):
            final_name = contract.model_artifact_set_sha256
            if _entry_exists(sha256_fd, final_name):
                try:
                    _verify_model_directory(sha256_fd, final_name, contract.components)
                except _PrepareError as error:
                    return _failure(error)
                return _ready(request, contract)

            return _stage_and_publish(request, contract, sha256_fd)
    except _PrepareError as error:
        return _failure(error)


def _verify_existing_cache(
    request: PrepareBackendRequest,
    contract: _LockContract,
    sha256_fd: int,
) -> PrepareBackendOutcome:
    final_name = contract.model_artifact_set_sha256
    if not _entry_exists(sha256_fd, final_name):
        return _failure(_AcquisitionError("checkpoint cache is missing"))
    try:
        _verify_model_directory(sha256_fd, final_name, contract.components)
    except _PrepareError as error:
        return _failure(error)
    return _ready(request, contract)


# The transaction keeps cleanup and publication ownership visible in one control flow.
# pylint: disable-next=too-many-branches,too-many-statements
def _stage_and_publish(
    request: PrepareBackendRequest,
    contract: _LockContract,
    sha256_fd: int,
) -> PrepareBackendOutcome:
    stage_name: str | None = None
    stage_fd: int | None = None
    try:
        stage_name, stage_fd = _create_staging_directory(
            sha256_fd,
            contract.model_artifact_set_sha256,
        )

        if request.download:
            archive_fd = _download_archive(stage_fd, contract)
        else:
            assert request.archive_path is not None
            archive_fd = _snapshot_local_archive(
                stage_fd,
                request.archive_path,
                contract,
            )

        try:
            _verify_archive_identity(archive_fd, contract)
            _extract_archive(archive_fd, stage_fd, contract)
            _verify_archive_identity(archive_fd, contract)
        finally:
            os.close(archive_fd)

        if not _remove_download_archive(stage_fd, ".checkpoint-archive"):
            raise _AcquisitionError("archive snapshot cleanup failed")
        if not _fsync_staging_directory(stage_fd):
            raise _AcquisitionError("staging directory synchronization failed")
        os.close(stage_fd)
        stage_fd = None

        _verify_model_directory(sha256_fd, stage_name, contract.components)
        try:
            rename_directory_no_replace(
                request.cache_root / "sha256" / stage_name,
                request.cache_root / "sha256" / contract.model_artifact_set_sha256,
            )
        except DirectoryPublicationError:
            if not _rollback_owned_directory(
                sha256_fd,
                contract.model_artifact_set_sha256,
            ):
                stage_name = None
                raise _IntegrityError("published checkpoint rollback failed") from None
            stage_name = None
            raise _IntegrityError("atomic checkpoint publication failed") from None
        except ArtifactAlreadyPublishedError:
            if not _cleanup_staging_directory(sha256_fd, stage_name):
                raise _AcquisitionError("losing staging cleanup failed") from None
            stage_name = None
            _verify_model_directory(
                sha256_fd,
                contract.model_artifact_set_sha256,
                contract.components,
            )
            return _ready(request, contract)
        except OSError:
            raise _IntegrityError("atomic checkpoint publication failed") from None

        stage_name = None
        try:
            os.fsync(sha256_fd)
            _verify_model_directory(
                sha256_fd,
                contract.model_artifact_set_sha256,
                contract.components,
            )
        except (OSError, _PrepareError):
            _rollback_owned_directory(
                sha256_fd,
                contract.model_artifact_set_sha256,
            )
            raise _IntegrityError("published checkpoint failed final verification") from None
        return _ready(request, contract)
    except _PrepareError as error:
        if stage_fd is not None:
            os.close(stage_fd)
            stage_fd = None
        if stage_name is not None and not _cleanup_staging_directory(sha256_fd, stage_name):
            if isinstance(error, _IntegrityError):
                return _failure(error)
            return _failure(_AcquisitionError("staging cleanup failed"))
        return _failure(error)
    except OSError:
        if stage_fd is not None:
            os.close(stage_fd)
        if stage_name is not None:
            _cleanup_staging_directory(sha256_fd, stage_name)
        return _failure(_AcquisitionError("local checkpoint staging failed"))


def _contract_from_validated_lock(
    request: PrepareBackendRequest,
    backend_lock: LoadedBackendLock,
) -> _LockContract | _PrepareError:
    payload = backend_lock.payload
    if payload.get("backend_id") != request.backend_id:
        return _IntegrityError("backend selection does not match the loaded lock")
    archive = cast(Mapping[str, object], payload["checkpoint_archive"])
    component_values = cast(Sequence[Mapping[str, object]], payload["checkpoint_components"])
    components = tuple(
        _Component(
            name=cast(str, row["name"]),
            size=cast(int, row["size"]),
            sha256=cast(str, row["sha256"]),
        )
        for row in component_values
    )
    return _LockContract(
        archive_size=cast(int, archive["size"]),
        archive_sha256=cast(str, archive["sha256"]),
        checkpoint_url=cast(str, payload["checkpoint_url"]),
        components=components,
        model_artifact_set_sha256=cast(
            str,
            backend_lock.descriptor.payload["model_artifact_set_sha256"],
        ),
    )


def _contract_from_acquisition_request(
    request: PrepareBackendRequest,
    acquisition_request: CheckpointAcquisitionRequest,
) -> _LockContract:
    if acquisition_request.backend_id != request.backend_id:
        raise CheckpointAcquisitionError("checkpoint acquisition backend does not match selection")
    components = tuple(
        _Component(member.name, member.size, member.sha256)
        for member in acquisition_request.archive_members
        if member.role == "published_component"
    )
    model_artifact_set_sha256 = sha256_hex(
        canonical_json_bytes(
            [
                {"name": component.name, "sha256": component.sha256, "size": component.size}
                for component in components
            ]
        )
    )
    return _LockContract(
        archive_size=acquisition_request.archive.size,
        archive_sha256=acquisition_request.archive.sha256,
        checkpoint_url=acquisition_request.checkpoint_url,
        components=components,
        model_artifact_set_sha256=model_artifact_set_sha256,
        archive_members=tuple(
            _Component(member.name, member.size, member.sha256)
            for member in acquisition_request.archive_members
        ),
        pointer=next(
            _Component(member.name, member.size, member.sha256)
            for member in acquisition_request.archive_members
            if member.role == "pointer"
        ),
    )


def _contract_identity(contract: _LockContract) -> tuple[object, ...]:
    return (
        contract.archive_size,
        contract.archive_sha256,
        contract.checkpoint_url,
        tuple(
            (component.name, component.sha256, component.size) for component in contract.components
        ),
        contract.model_artifact_set_sha256,
    )


def _failure(error: _PrepareError) -> PrepareBackendOutcome:
    return PrepareBackendOutcome(
        status=error.status,
        exit_code=error.exit_code,
        model_cache_path=None,
    )


def _ready(
    request: PrepareBackendRequest,
    contract: _LockContract,
) -> PrepareBackendOutcome:
    return PrepareBackendOutcome(
        status="ready",
        exit_code=0,
        model_cache_path=(request.cache_root / "sha256" / contract.model_artifact_set_sha256),
    )


def _open_directory_chain(path: Path, *, create: bool) -> int:
    absolute = Path(os.path.abspath(path))
    current_fd = os.open("/", os.O_RDONLY | _DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            try:
                child_fd = os.open(
                    component,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                    dir_fd=current_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise _AcquisitionError("cache directory is missing") from None
                try:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                    else:
                        os.fsync(current_fd)
                    child_fd = os.open(
                        component,
                        os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                        dir_fd=current_fd,
                    )
                except OSError as error:
                    raise _classify_directory_error(error) from None
            except OSError as error:
                raise _classify_directory_error(error) from None
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _open_child_directory(parent_fd: int, name: str, *, create: bool) -> int:
    try:
        return os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise _AcquisitionError("cache directory is missing") from None
        try:
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            else:
                os.fsync(parent_fd)
            return os.open(
                name,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise _classify_directory_error(error) from None
    except OSError as error:
        raise _classify_directory_error(error) from None


def _classify_directory_error(error: OSError) -> _PrepareError:
    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
        return _IntegrityError("cache ancestry is not a regular directory")
    return _AcquisitionError("cache directory could not be prepared")


@contextmanager
def _publication_lock(sha256_fd: int):
    try:
        lock_fd = _open_publication_lock(sha256_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise _IntegrityError("cache publication lock is unsafe") from None
        raise _AcquisitionError("cache publication lock is unavailable") from None
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise _IntegrityError("cache publication lock is unsafe")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        except OSError:
            raise _AcquisitionError("cache publication lock is unavailable") from None
        yield
    finally:
        os.close(lock_fd)


def _open_publication_lock(sha256_fd: int) -> int:
    name = ".prepare-backend.lock"
    try:
        return os.open(name, os.O_RDWR | _NOFOLLOW, dir_fd=sha256_fd)
    except FileNotFoundError:
        try:
            lock_fd = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                0o600,
                dir_fd=sha256_fd,
            )
        except FileExistsError:
            return os.open(name, os.O_RDWR | _NOFOLLOW, dir_fd=sha256_fd)
        os.fsync(sha256_fd)
        return lock_fd


def _entry_exists(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _create_staging_directory(parent_fd: int, digest: str) -> tuple[str, int]:
    for _ in range(16):
        name = f".{digest}.staging-{secrets.token_hex(8)}"
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            fd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=parent_fd)
            return name, fd
        except FileExistsError:
            continue
        except OSError:
            raise _AcquisitionError("private checkpoint staging could not be created") from None
    raise _AcquisitionError("private checkpoint staging name could not be reserved")


def _snapshot_local_archive(
    stage_fd: int,
    path: Path,
    contract: _LockContract,
) -> int:
    source_fd = _open_local_archive(path)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW
    try:
        snapshot_fd = os.open(".checkpoint-archive", flags, 0o600, dir_fd=stage_fd)
    except OSError:
        os.close(source_fd)
        raise _AcquisitionError("archive snapshot could not be created") from None
    try:
        try:
            total = 0
            while True:
                content = os.read(
                    source_fd,
                    min(_READ_CHUNK_BYTES, contract.archive_size - total + 1),
                )
                if not content:
                    break
                total += len(content)
                if total > contract.archive_size:
                    raise _IntegrityError("supplied checkpoint archive is oversized")
                _write_all(snapshot_fd, content)
            os.fsync(snapshot_fd)
            os.lseek(snapshot_fd, 0, os.SEEK_SET)
        except OSError:
            raise _AcquisitionError("archive snapshot failed") from None
        return snapshot_fd
    except BaseException:
        os.close(snapshot_fd)
        raise
    finally:
        os.close(source_fd)


def _open_local_archive(path: Path) -> int:
    absolute = Path(os.path.abspath(path))
    if len(absolute.parts) < 2:
        raise _IntegrityError("supplied checkpoint archive path is invalid")
    current_fd = os.open("/", os.O_RDONLY | _DIRECTORY)
    try:
        for component in absolute.parts[1:-1]:
            try:
                child_fd = os.open(
                    component,
                    os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                    dir_fd=current_fd,
                )
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise _IntegrityError("supplied archive ancestry is unsafe") from None
                raise _AcquisitionError("supplied archive ancestry is unavailable") from None
            os.close(current_fd)
            current_fd = child_fd
        try:
            archive_fd = os.open(
                absolute.parts[-1],
                os.O_RDONLY | _NOFOLLOW,
                dir_fd=current_fd,
            )
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise _IntegrityError("supplied checkpoint archive is unsafe") from None
            raise _AcquisitionError("supplied checkpoint archive is unavailable") from None
        if not stat.S_ISREG(os.fstat(archive_fd).st_mode):
            os.close(archive_fd)
            raise _IntegrityError("supplied checkpoint archive is not a regular file")
        return archive_fd
    finally:
        os.close(current_fd)


def _download_archive(stage_fd: int, contract: _LockContract) -> int:
    _validate_download_url(contract.checkpoint_url)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW
    try:
        archive_fd = os.open(".checkpoint-archive", flags, 0o600, dir_fd=stage_fd)
    except OSError:
        raise _AcquisitionError("download staging file could not be created") from None
    try:
        try:
            response = _open_download_url(contract.checkpoint_url)
            with response:
                effective_url = response.geturl()
                if effective_url != contract.checkpoint_url:
                    raise _IntegrityError("checkpoint redirect or URL substitution rejected")
                total = 0
                while True:
                    content = response.read(
                        min(_READ_CHUNK_BYTES, contract.archive_size - total + 1)
                    )
                    if not content:
                        break
                    total += len(content)
                    if total > contract.archive_size:
                        raise _IntegrityError("downloaded checkpoint archive is oversized")
                    _write_all(archive_fd, content)
        except (OSError, ValueError):
            raise _AcquisitionError("checkpoint download failed") from None
        try:
            os.fsync(archive_fd)
            os.lseek(archive_fd, 0, os.SEEK_SET)
        except OSError:
            raise _AcquisitionError("download staging synchronization failed") from None
        return archive_fd
    except BaseException:
        os.close(archive_fd)
        raise


def _validate_download_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
    except ValueError:
        raise _IntegrityError("checkpoint URL is invalid") from None
    if url != _OFFICIAL_CHECKPOINT_URL:
        raise _IntegrityError("checkpoint URL is outside the exact allowlist")
    if parsed.scheme != "https" or parsed.hostname != "storage.googleapis.com":
        raise _IntegrityError("checkpoint URL is outside the exact allowlist")
    if parsed.username is not None or parsed.password is not None:
        raise _IntegrityError("checkpoint URL is outside the exact allowlist")
    if parsed.port is not None or parsed.query or parsed.fragment:
        raise _IntegrityError("checkpoint URL is outside the exact allowlist")


def _open_download_url(url: str) -> BinaryIO:
    return cast(BinaryIO, urllib.request.build_opener(_RejectRedirects()).open(url))


def _verify_archive_identity(archive_fd: int, contract: _LockContract) -> None:
    metadata = os.fstat(archive_fd)
    if not stat.S_ISREG(metadata.st_mode):
        raise _IntegrityError("checkpoint archive is not a regular file")
    if metadata.st_size != contract.archive_size:
        raise _IntegrityError("checkpoint archive size contradicts the lock")
    digest = hashlib.sha256()
    try:
        os.lseek(archive_fd, 0, os.SEEK_SET)
        while True:
            content = os.read(archive_fd, _READ_CHUNK_BYTES)
            if not content:
                break
            digest.update(content)
        os.lseek(archive_fd, 0, os.SEEK_SET)
    except OSError:
        raise _AcquisitionError("checkpoint archive could not be read") from None
    if digest.hexdigest() != contract.archive_sha256:
        raise _IntegrityError("checkpoint archive hash contradicts the lock")


def _extract_archive(
    archive_fd: int,
    stage_fd: int,
    contract: _LockContract,
) -> None:
    members = _parse_zip_members(archive_fd, contract)
    if contract.pointer is not None:
        _verify_pointer_member(archive_fd, members[contract.pointer.name], contract.pointer)
    for component in contract.components:
        _extract_component(
            archive_fd,
            members[component.name],
            stage_fd,
            component,
        )


# Central-directory records expose many independent security-relevant fields.
# pylint: disable-next=too-many-locals
def _parse_zip_members(
    archive_fd: int,
    contract: _LockContract,
) -> dict[str, _ZipMember]:
    central_offset, central_size, member_count = _parse_eocd(archive_fd, contract.archive_size)
    expected_members = contract.archive_members or contract.components
    expected = {component.name: component for component in expected_members}
    if member_count != len(expected):
        raise _IntegrityError("checkpoint archive member set is incomplete")

    members: dict[str, _ZipMember] = {}
    cursor = central_offset
    central_end = central_offset + central_size
    for _ in range(member_count):
        fixed = _read_archive_bytes(archive_fd, cursor, _CENTRAL_HEADER.size)
        values = _CENTRAL_HEADER.unpack(fixed)
        if values[0] != _CENTRAL_SIGNATURE:
            raise _IntegrityError("checkpoint archive central directory is invalid")
        (
            _,
            version_made,
            version_needed,
            flags,
            compression,
            _modified_time,
            _modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            disk_start,
            _internal_attributes,
            external_attributes,
            local_header_offset,
        ) = values
        variable_size = name_length + extra_length + comment_length
        variable = _read_archive_bytes(
            archive_fd,
            cursor + _CENTRAL_HEADER.size,
            variable_size,
        )
        cursor += _CENTRAL_HEADER.size + variable_size
        if cursor > central_end:
            raise _IntegrityError("checkpoint archive central directory is invalid")
        raw_name = variable[:name_length]
        extra = variable[name_length : name_length + extra_length]
        _validate_extra_fields(extra)
        _reject_zip64_values(
            version_needed,
            compressed_size,
            uncompressed_size,
            local_header_offset,
            disk_start,
        )
        name = _decode_member_name(raw_name, flags)
        if not _is_safe_basename(name) or name not in expected or name in members:
            raise _IntegrityError("checkpoint archive member set is unsafe")
        _validate_member_metadata(
            version_made,
            flags,
            compression,
            external_attributes,
        )
        component = expected[name]
        if uncompressed_size != component.size:
            raise _IntegrityError("checkpoint archive member size is suspicious")
        members[name] = _parse_local_member(
            archive_fd,
            name=name,
            raw_name=raw_name,
            flags=flags,
            compression=compression,
            crc32=crc32,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            local_header_offset=local_header_offset,
            central_offset=central_offset,
        )

    if cursor != central_end or set(members) != set(expected):
        raise _IntegrityError("checkpoint archive member set is incomplete")
    regions = sorted((member.local_header_offset, member.region_end) for member in members.values())
    if any(end > next_start for (_, end), (next_start, _) in zip(regions, regions[1:])):
        raise _IntegrityError("checkpoint archive member regions overlap")
    return members


def _verify_pointer_member(
    archive_fd: int,
    member: _ZipMember,
    pointer: _Component,
) -> None:
    try:
        compressed = _read_archive_bytes(archive_fd, member.data_offset, member.compressed_size)
        if member.compression == zipfile.ZIP_DEFLATED:
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            content = decompressor.decompress(compressed, pointer.size + 1)
            content += decompressor.flush(pointer.size + 1 - len(content))
            if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
                raise _IntegrityError("checkpoint pointer stream is invalid")
        else:
            content = compressed
    except zlib.error:
        raise _IntegrityError("checkpoint pointer stream is invalid") from None
    expected = (
        b'model_checkpoint_path: "model.ckpt-569400"\n'
        b'all_model_checkpoint_paths: "model.ckpt-569400"\n'
    )
    if (
        content != expected
        or len(content) != pointer.size
        or hashlib.sha256(content).hexdigest() != pointer.sha256
        or zlib.crc32(content) != member.crc32
    ):
        raise _IntegrityError("checkpoint pointer payload contradicts the request")


# EOCD search keeps candidate and validated metadata separate.
# pylint: disable-next=too-many-locals
def _parse_eocd(archive_fd: int, archive_size: int) -> tuple[int, int, int]:
    maximum_eocd_size = _EOCD.size + 0xFFFF
    tail_offset = max(0, archive_size - maximum_eocd_size)
    tail = _read_archive_bytes(archive_fd, tail_offset, archive_size - tail_offset)
    search_end = len(tail)
    while True:
        relative_offset = tail.rfind(_EOCD_SIGNATURE, 0, search_end)
        if relative_offset < 0:
            raise _IntegrityError("checkpoint archive end record is invalid")
        if relative_offset + _EOCD.size <= len(tail):
            values = _EOCD.unpack_from(tail, relative_offset)
            comment_length = values[-1]
            if relative_offset + _EOCD.size + comment_length == len(tail):
                break
        search_end = relative_offset

    (
        _,
        disk_number,
        central_disk,
        entries_on_disk,
        entry_count,
        central_size,
        central_offset,
        _,
    ) = values
    invalid_disk_layout = disk_number != 0 or central_disk != 0
    invalid_entry_count = entries_on_disk != entry_count or entry_count == 0xFFFF
    zip64_sentinel = _ZIP32_SENTINEL in (central_size, central_offset)
    invalid_boundary = central_offset + central_size != tail_offset + relative_offset
    if invalid_disk_layout or invalid_entry_count or zip64_sentinel or invalid_boundary:
        raise _IntegrityError("checkpoint archive end record is unsupported")
    return central_offset, central_size, entry_count


def _validate_extra_fields(extra: bytes) -> None:
    cursor = 0
    while cursor < len(extra):
        if len(extra) - cursor < 4:
            raise _IntegrityError("checkpoint archive extra field is invalid")
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        if cursor + field_size > len(extra):
            raise _IntegrityError("checkpoint archive extra field is invalid")
        if field_id == _ZIP64_EXTRA_ID:
            # The frozen v1 cache contract is deliberately ZIP32-only. Supporting
            # ZIP64 would add a second metadata authority for sizes and offsets.
            raise _IntegrityError("ZIP64 checkpoint archives are unsupported")
        cursor += field_size


def _reject_zip64_values(
    version_needed: int,
    compressed_size: int,
    uncompressed_size: int,
    local_header_offset: int,
    disk_start: int,
) -> None:
    if disk_start != 0:
        raise _IntegrityError("multi-disk checkpoint archives are unsupported")
    if (
        version_needed >= 45
        or compressed_size == _ZIP32_SENTINEL
        or uncompressed_size == _ZIP32_SENTINEL
        or local_header_offset == _ZIP32_SENTINEL
    ):
        raise _IntegrityError("ZIP64 checkpoint archives are unsupported")


def _decode_member_name(raw_name: bytes, flags: int) -> str:
    try:
        return raw_name.decode("utf-8" if flags & 0x0800 else "cp437")
    except UnicodeDecodeError:
        raise _IntegrityError("checkpoint archive member name is invalid") from None


def _validate_member_metadata(
    version_made: int,
    flags: int,
    compression: int,
    external_attributes: int,
) -> None:
    if flags & ~_ALLOWED_GENERAL_FLAGS or flags & 0x0001:
        raise _IntegrityError("checkpoint archive member flags are unsupported")
    if flags & 0x0006 == 0x0006:
        raise _IntegrityError("checkpoint archive deflate flags are invalid")
    if compression not in _SUPPORTED_COMPRESSION:
        raise _IntegrityError("checkpoint archive compression is unsupported")
    if compression == zipfile.ZIP_STORED and flags & 0x0006:
        raise _IntegrityError("stored checkpoint member has deflate flags")
    if external_attributes & 0x10:
        raise _IntegrityError("checkpoint archive member has DOS directory attributes")
    if version_made >> 8 == 3:
        unix_mode = (external_attributes >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type and not stat.S_ISREG(unix_mode):
            raise _IntegrityError("checkpoint archive member is not regular")


# The local header must be compared field-for-field with its central record.
# pylint: disable-next=too-many-arguments,too-many-locals
def _parse_local_member(
    archive_fd: int,
    *,
    name: str,
    raw_name: bytes,
    flags: int,
    compression: int,
    crc32: int,
    compressed_size: int,
    uncompressed_size: int,
    local_header_offset: int,
    central_offset: int,
) -> _ZipMember:
    fixed = _read_archive_bytes(archive_fd, local_header_offset, _LOCAL_HEADER.size)
    values = _LOCAL_HEADER.unpack(fixed)
    if values[0] != _LOCAL_SIGNATURE:
        raise _IntegrityError("checkpoint archive local header is invalid")
    (
        _,
        version_needed,
        local_flags,
        local_compression,
        _modified_time,
        _modified_date,
        local_crc32,
        local_compressed_size,
        local_uncompressed_size,
        name_length,
        extra_length,
    ) = values
    _reject_zip64_values(
        version_needed,
        local_compressed_size,
        local_uncompressed_size,
        local_header_offset,
        0,
    )
    variable = _read_archive_bytes(
        archive_fd,
        local_header_offset + _LOCAL_HEADER.size,
        name_length + extra_length,
    )
    local_name = variable[:name_length]
    _validate_extra_fields(variable[name_length:])
    if local_name != raw_name or local_flags != flags or local_compression != compression:
        raise _IntegrityError("checkpoint archive headers disagree")
    descriptor_used = bool(flags & 0x0008)
    if descriptor_used:
        local_values = (local_crc32, local_compressed_size, local_uncompressed_size)
        if local_values not in {
            (0, 0, 0),
            (crc32, compressed_size, uncompressed_size),
        }:
            raise _IntegrityError("checkpoint archive descriptor metadata disagrees")
    elif (
        local_crc32,
        local_compressed_size,
        local_uncompressed_size,
    ) != (crc32, compressed_size, uncompressed_size):
        raise _IntegrityError("checkpoint archive headers disagree")

    if compression == zipfile.ZIP_STORED and compressed_size != uncompressed_size:
        raise _IntegrityError("stored checkpoint member sizes disagree")
    data_offset = local_header_offset + _LOCAL_HEADER.size + name_length + extra_length
    region_end = data_offset + compressed_size
    if descriptor_used:
        descriptor = _read_archive_bytes(archive_fd, region_end, 16)
        signature, descriptor_crc32, descriptor_compressed, descriptor_uncompressed = struct.unpack(
            "<4s3L", descriptor
        )
        if (
            signature != _DATA_DESCRIPTOR_SIGNATURE
            or descriptor_crc32 != crc32
            or descriptor_compressed != compressed_size
            or descriptor_uncompressed != uncompressed_size
        ):
            raise _IntegrityError("checkpoint archive data descriptor is invalid")
        region_end += 16
    if region_end > central_offset:
        raise _IntegrityError("checkpoint archive member extends into metadata")
    return _ZipMember(
        name=name,
        raw_name=raw_name,
        flags=flags,
        compression=compression,
        crc32=crc32,
        compressed_size=compressed_size,
        uncompressed_size=uncompressed_size,
        local_header_offset=local_header_offset,
        data_offset=data_offset,
        region_end=region_end,
    )


def _read_archive_bytes(archive_fd: int, offset: int, length: int) -> bytes:
    if offset < 0 or length < 0:
        raise _IntegrityError("checkpoint archive offset is invalid")
    chunks: list[bytes] = []
    remaining = length
    try:
        while remaining:
            content = os.pread(
                archive_fd,
                min(_READ_CHUNK_BYTES, remaining),
                offset + length - remaining,
            )
            if not content:
                raise _IntegrityError("checkpoint archive is truncated")
            chunks.append(content)
            remaining -= len(content)
    except OSError:
        raise _AcquisitionError("checkpoint archive could not be read") from None
    return b"".join(chunks)


def _is_safe_basename(name: str) -> bool:
    return (
        bool(name)
        and "\x00" not in name
        and "/" not in name
        and "\\" not in name
        and ":" not in name
        and name not in {".", ".."}
        and Path(name).name == name
    )


def _extract_component(
    archive_fd: int,
    member: _ZipMember,
    stage_fd: int,
    component: _Component,
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
    try:
        output_fd = os.open(component.name, flags, 0o600, dir_fd=stage_fd)
    except OSError:
        raise _AcquisitionError("checkpoint component staging failed") from None
    digest = hashlib.sha256()
    size = 0
    checksum = 0
    try:
        try:
            decompressor = (
                zlib.decompressobj(-zlib.MAX_WBITS)
                if member.compression == zipfile.ZIP_DEFLATED
                else None
            )
            compressed_offset = 0
            while compressed_offset < member.compressed_size:
                compressed = _read_archive_bytes(
                    archive_fd,
                    member.data_offset + compressed_offset,
                    min(_READ_CHUNK_BYTES, member.compressed_size - compressed_offset),
                )
                compressed_offset += len(compressed)
                if decompressor is None:
                    content = compressed
                else:
                    content = decompressor.decompress(
                        compressed,
                        component.size - size + 1,
                    )
                    if decompressor.unconsumed_tail:
                        raise _IntegrityError("checkpoint component decompressed past lock")
                size, checksum = _write_component_content(
                    output_fd,
                    content,
                    digest,
                    size,
                    checksum,
                    component.size,
                )
            if decompressor is not None:
                content = decompressor.flush(component.size - size + 1)
                size, checksum = _write_component_content(
                    output_fd,
                    content,
                    digest,
                    size,
                    checksum,
                    component.size,
                )
                if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
                    raise _IntegrityError("checkpoint component stream is invalid")
            if (
                size != component.size
                or digest.hexdigest() != component.sha256
                or checksum != member.crc32
            ):
                raise _IntegrityError("checkpoint component bytes contradict the lock")
            os.fsync(output_fd)
        except zlib.error:
            raise _IntegrityError("checkpoint component stream is invalid") from None
        except OSError:
            raise _AcquisitionError("checkpoint component staging failed") from None
    finally:
        os.close(output_fd)


# Keeping stream accounting together avoids partial writes before limit checks.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _write_component_content(
    output_fd: int,
    content: bytes,
    digest: _Digest,
    size: int,
    checksum: int,
    maximum_size: int,
) -> tuple[int, int]:
    next_size = size + len(content)
    if next_size > maximum_size:
        raise _IntegrityError("checkpoint component decompressed past lock")
    digest.update(content)
    checksum = zlib.crc32(content, checksum)
    _write_all(output_fd, content)
    return next_size, checksum


def _write_all(fd: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short checkpoint write")
        view = view[written:]


def _remove_download_archive(stage_fd: int, name: str) -> bool:
    try:
        metadata = os.stat(name, dir_fd=stage_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            return False
        os.unlink(name, dir_fd=stage_fd)
        os.fsync(stage_fd)
        return True
    except OSError:
        return False


def _fsync_staging_directory(stage_fd: int) -> bool:
    try:
        os.fsync(stage_fd)
        return True
    except OSError:
        return False


def _verify_model_directory(
    parent_fd: int,
    name: str,
    components: tuple[_Component, ...],
) -> None:
    try:
        directory_fd = os.open(
            name,
            os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError:
        raise _IntegrityError("checkpoint cache directory is unsafe") from None
    try:
        if set(os.listdir(directory_fd)) != {component.name for component in components}:
            raise _IntegrityError("checkpoint cache entries contradict the lock")
        for component in components:
            _verify_component_file(directory_fd, component)
    finally:
        os.close(directory_fd)


def _verify_component_file(directory_fd: int, component: _Component) -> None:
    try:
        file_fd = os.open(component.name, os.O_RDONLY | _NOFOLLOW, dir_fd=directory_fd)
    except OSError:
        raise _IntegrityError("checkpoint cache component is unsafe") from None
    try:
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != component.size:
            raise _IntegrityError("checkpoint cache component size contradicts the lock")
        digest = hashlib.sha256()
        total = 0
        while True:
            content = os.read(file_fd, min(_READ_CHUNK_BYTES, component.size - total + 1))
            if not content:
                break
            total += len(content)
            if total > component.size:
                raise _IntegrityError("checkpoint cache component is oversized")
            digest.update(content)
        if total != component.size or digest.hexdigest() != component.sha256:
            raise _IntegrityError("checkpoint cache component hash contradicts the lock")
    except OSError:
        raise _IntegrityError("checkpoint cache component could not be verified") from None
    finally:
        os.close(file_fd)


def _cleanup_staging_directory(parent_fd: int, name: str) -> bool:
    try:
        directory_fd = os.open(
            name,
            os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
            dir_fd=parent_fd,
        )
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        for entry in os.listdir(directory_fd):
            metadata = os.stat(entry, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                return False
            os.unlink(entry, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError:
        return False
    finally:
        os.close(directory_fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    except OSError:
        return False


def _rollback_owned_directory(parent_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return _cleanup_staging_directory(parent_fd, name)
