from __future__ import annotations

import os
import stat
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import ContextManager
from uuid import UUID, uuid4

from src.benchmark.corpus_cache import (
    CacheIndexStore,
    cache_writer_lock,
    is_selected,
    sync_cache,
)
from src.benchmark.corpus_manifest import (
    ManifestPublicationError,
    build_manifest_rows,
    canonical_json_line,
    publish_latest_manifest,
    publish_manifest,
    render_manifest,
)
from src.benchmark.corpus_provenance import load_provenance
from src.benchmark.r2_corpus_models import (
    CACHE_PROFILE,
    REPORT_SCHEMA,
    CacheAction,
    CacheSyncResult,
    ErrorCode,
    InventoryResult,
    OverallStatus,
    ProvenanceRecord,
    PublishedManifest,
    R2Config,
    SyncCounters,
    SyncError,
    SyncOutcome,
    SyncRequest,
    format_manifest_timestamp,
    format_report_filename_timestamp,
)
from src.benchmark.r2_inventory import (
    R2ObjectStore,
    R2StoreError,
    build_inventory,
    create_boto3_store,
    ensure_r2_dependency,
)


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    completed: int
    total: int | None
    message: str


ProgressCallback = Callable[[ProgressEvent], None]
StoreFactory = Callable[[R2Config], R2ObjectStore]
Clock = Callable[[], datetime]
MonotonicClock = Callable[[], float]
RunIdFactory = Callable[[], str]

_SAFE_MESSAGES: dict[ErrorCode, str] = {
    "invalid_config": "R2 synchronization configuration is invalid.",
    "missing_optional_dependency": (
        "R2 support is not installed. Install it with uv pip install -e '.[r2]'."
    ),
    "missing_credentials": "No usable R2 credentials were resolved.",
    "auth_failed": "R2 rejected the resolved credentials.",
    "bucket_inaccessible": "The configured R2 bucket is inaccessible.",
    "root_list_failed": "R2 root listing did not complete.",
    "cache_locked": "Another synchronization writer holds the cache lock.",
    "cache_index_invalid": "The cache index is malformed or unsupported.",
    "unsupported_platform": "Required POSIX locking or durability support is unavailable.",
    "provenance_invalid": "The provenance document is malformed or unsupported.",
    "artifact_write_failed": "A required synchronization artifact write failed.",
    "object_head_failed": "Object metadata inspection failed.",
    "object_get_failed": "Object body read failed.",
    "source_changed_during_sync": "Object metadata changed after inventory.",
    "weak_etag_unverifiable": "Weak ETag response metadata cannot verify the inventory.",
    "byte_count_mismatch": "Downloaded byte count does not match object metadata.",
    "cache_corrupt": "A referenced cache body is corrupt and could not be repaired.",
    "object_metadata_invalid": "Object metadata is invalid.",
    "ambiguous_simfile_prefix": "Multiple exact prefixes normalize to one numeric ID.",
    "malformed_root_key": "A root key cannot be assigned to a valid simfile prefix.",
    "empty_prefix": "A requested prefix has no objects or only folder markers.",
    "internal_error": "R2 synchronization failed because of an internal error.",
}
_ADAPTER_CODES = frozenset(
    {
        "missing_optional_dependency",
        "missing_credentials",
        "auth_failed",
        "bucket_inaccessible",
        "root_list_failed",
        "object_head_failed",
        "object_get_failed",
        "source_changed_during_sync",
        "weak_etag_unverifiable",
        "byte_count_mismatch",
        "object_metadata_invalid",
    }
)
_CACHE_MISS_REASONS = (
    "remote_changed",
    "missing",
    "size_mismatch",
    "sha256_mismatch",
)
_publication_lock = Lock()


class _SyncFailure(Exception):
    def __init__(self, error: SyncError):
        super().__init__(error.code)
        self.error = error


class _ReportWriteFailure(Exception):
    pass


@dataclass
class _RunState:
    config: R2Config | None = None
    provenance: Mapping[int, ProvenanceRecord] | None = None
    inventory: InventoryResult | None = None
    cache_result: CacheSyncResult | None = None
    counters: SyncCounters = SyncCounters()
    immutable_manifest: PublishedManifest | None = None


class _Progress:
    def __init__(self, callback: ProgressCallback, monotonic: MonotonicClock) -> None:
        self._callback = callback
        self._monotonic = monotonic
        self._entered: set[str] = set()

    def enter(self, phase: str, total: int | None = None) -> None:
        if phase in self._entered:
            return
        self._entered.add(phase)
        self._callback(ProgressEvent(phase, 0, total, f"{phase} phase started."))

    def items(
        self,
        phase: str,
        totals: tuple[int, ...],
        *,
        unit: str,
    ) -> None:
        total = len(totals)
        if total == 0:
            return
        last_event_at = self._monotonic()
        accumulated = 0
        for completed, amount in enumerate(totals, start=1):
            accumulated += amount
            now = self._monotonic()
            if completed % 100 == 0 or completed == total or now - last_event_at >= 5:
                self._callback(
                    ProgressEvent(
                        phase,
                        completed,
                        total,
                        f"{phase}: {completed}/{total} {unit}, {accumulated} bytes.",
                    )
                )
                last_event_at = now

    def finish(self, status: OverallStatus) -> None:
        self._callback(ProgressEvent(status, 1, 1, f"{status} synchronization outcome."))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_run_id() -> str:
    return str(uuid4())


def ignore_progress(_event: ProgressEvent) -> None:
    return None


def report_filename(started_at: datetime, run_id: str) -> str:
    _validate_aware_datetime(started_at)
    _validate_run_id(run_id)
    stamp = format_report_filename_timestamp(started_at)
    return f"{stamp}-{run_id}.json"


def sync_r2_corpus(
    request: SyncRequest,
    *,
    environ: Mapping[str, str] | None = None,
    dependency_check: Callable[[], None] = ensure_r2_dependency,
    store_factory: StoreFactory = create_boto3_store,
    clock: Clock = utc_now,
    monotonic: MonotonicClock = time.monotonic,
    run_id_factory: RunIdFactory = new_run_id,
    progress: ProgressCallback = ignore_progress,
) -> SyncOutcome:
    try:
        started_at = clock()
        run_id = run_id_factory()
    except Exception:
        started_at = utc_now()
        run_id = new_run_id()
        tracker = _Progress(progress, monotonic)
        tracker.enter("configuration")
        return _fatal_outcome(
            request=request,
            run_id=run_id,
            started_at=started_at,
            clock=clock,
            tracker=tracker,
            state=_RunState(),
            error=_safe_error("internal_error"),
        )
    return _run_sync(
        request=request,
        environ=os.environ if environ is None else environ,
        dependency_check=dependency_check,
        store_factory=store_factory,
        clock=clock,
        run_id=run_id,
        started_at=started_at,
        progress=progress,
        monotonic=monotonic,
    )


def _run_sync(
    *,
    request: SyncRequest,
    environ: Mapping[str, str],
    dependency_check: Callable[[], None],
    store_factory: StoreFactory,
    clock: Clock,
    run_id: str,
    started_at: datetime,
    progress: ProgressCallback,
    monotonic: MonotonicClock,
) -> SyncOutcome:
    state = _RunState()
    tracker = _Progress(progress, monotonic)
    tracker.enter("configuration")
    try:
        dependency_check()
        _validate_aware_datetime(started_at)
        _validate_run_id(run_id)
        state.config = R2Config.from_environ(environ)
        _validate_request_paths(request)
        try:
            state.provenance = load_provenance(request.provenance_file)
        except ValueError:
            raise _SyncFailure(_safe_error("provenance_invalid")) from None
    except R2StoreError as error:
        return _fatal_outcome(
            request, run_id, started_at, clock, tracker, state, _adapter_error(error)
        )
    except _SyncFailure as failure:
        return _fatal_outcome(request, run_id, started_at, clock, tracker, state, failure.error)
    except ValueError:
        return _fatal_outcome(
            request, run_id, started_at, clock, tracker, state, _safe_error("invalid_config")
        )
    except Exception:
        return _fatal_outcome(
            request, run_id, started_at, clock, tracker, state, _safe_error("internal_error")
        )

    lock: ContextManager[None]
    lock = nullcontext() if request.dry_run else cache_writer_lock(request.cache_dir)
    try:
        with lock:
            try:
                return _run_transaction(
                    request=request,
                    store_factory=store_factory,
                    clock=clock,
                    run_id=run_id,
                    started_at=started_at,
                    tracker=tracker,
                    state=state,
                )
            except _ReportWriteFailure:
                return _report_fallback_outcome(tracker)
            except R2StoreError as error:
                return _fatal_outcome(
                    request, run_id, started_at, clock, tracker, state, _adapter_error(error)
                )
            except ManifestPublicationError as error:
                return _fatal_outcome(
                    request, run_id, started_at, clock, tracker, state, error.error
                )
            except _SyncFailure as failure:
                return _fatal_outcome(
                    request, run_id, started_at, clock, tracker, state, failure.error
                )
            except Exception:
                return _fatal_outcome(
                    request,
                    run_id,
                    started_at,
                    clock,
                    tracker,
                    state,
                    _safe_error("internal_error"),
                )
    except RuntimeError as error:
        code = _lock_error_code(error)
        return _fatal_outcome(request, run_id, started_at, clock, tracker, state, _safe_error(code))
    except (OSError, ValueError):
        return _fatal_outcome(
            request,
            run_id,
            started_at,
            clock,
            tracker,
            state,
            _safe_error("artifact_write_failed"),
        )
    except Exception:
        return _fatal_outcome(
            request, run_id, started_at, clock, tracker, state, _safe_error("internal_error")
        )


def _run_transaction(
    *,
    request: SyncRequest,
    store_factory: StoreFactory,
    clock: Clock,
    run_id: str,
    started_at: datetime,
    tracker: _Progress,
    state: _RunState,
) -> SyncOutcome:
    assert state.config is not None
    assert state.provenance is not None
    try:
        index = CacheIndexStore.load(request.cache_dir)
    except ValueError:
        raise _SyncFailure(_safe_error("cache_index_invalid")) from None

    store = store_factory(state.config)
    tracker.enter("inventory")
    store.validate_bucket()
    tracker.enter("metadata")
    state.inventory = build_inventory(
        store,
        request.include_simfile_ids,
        request.exclude_simfile_ids,
        state.config.head_concurrency,
    )
    metadata_sizes = tuple(
        remote.size for simfile in state.inventory.simfiles for remote in simfile.objects
    )
    tracker.items("metadata", metadata_sizes, unit="objects")

    tracker.enter("cache")
    state.cache_result = sync_cache(
        state.inventory.simfiles,
        store,
        index,
        state.config,
        request.dry_run,
    )
    tracker.items(
        "cache",
        tuple(action.bytes for action in state.cache_result.actions),
        unit="objects",
    )
    state.counters = _build_counters(state.inventory, state.cache_result)
    is_partial = _is_partial(state.inventory, state.cache_result)
    overall_status: OverallStatus
    if request.dry_run:
        overall_status = "dry_run_partial" if is_partial else "dry_run_complete"
    else:
        overall_status = "partial" if is_partial else "complete"
    exit_code = 1 if is_partial else 0

    tracker.enter("manifest")
    if not request.dry_run:
        rows = build_manifest_rows(
            state.cache_result.simfiles,
            state.provenance,
            state.config.source_endpoint_sha256,
            state.config.bucket,
        )
        state.immutable_manifest = publish_manifest(request.output_dir, render_manifest(rows))

    tracker.enter("report")
    completed_at = clock()
    _validate_aware_datetime(completed_at)
    top_level_errors = state.inventory.root_errors
    report = _build_report(
        request=request,
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        overall_status=overall_status,
        exit_code=exit_code,
        state=state,
        top_level_errors=top_level_errors,
        manifest=state.immutable_manifest,
    )
    with _publication_lock:
        try:
            report_path, relative_report_path, report_sha256 = _publish_report_file(
                request.output_dir,
                started_at,
                run_id,
                report,
            )
        except Exception:
            raise _ReportWriteFailure from None
        previous_latest = None
        if state.immutable_manifest is not None:
            previous_latest = _snapshot_latest_manifest(request.output_dir)
            try:
                publish_latest_manifest(
                    request.output_dir,
                    state.immutable_manifest,
                    overall_status,
                    completed_at,
                )
            except Exception:
                try:
                    _restore_latest_manifest(request.output_dir, previous_latest)
                except Exception:
                    raise _ReportWriteFailure from None
                raise
        try:
            _publish_latest_report(
                request.output_dir,
                relative_report_path,
                overall_status,
                exit_code,
                (
                    state.immutable_manifest.corpus_version
                    if state.immutable_manifest is not None
                    else None
                ),
                report_sha256,
                completed_at,
            )
        except Exception:
            if state.immutable_manifest is not None:
                _restore_latest_manifest(request.output_dir, previous_latest)
            raise _ReportWriteFailure from None

    tracker.finish(overall_status)
    return SyncOutcome(
        overall_status=overall_status,
        exit_code=exit_code,
        report_path=report_path,
        manifest=state.immutable_manifest,
        errors=top_level_errors,
        counters=state.counters,
    )


def _fatal_outcome(
    request: SyncRequest,
    run_id: str,
    started_at: datetime,
    clock: Clock,
    tracker: _Progress,
    state: _RunState,
    error: SyncError,
) -> SyncOutcome:
    tracker.enter("report")
    try:
        completed_at = clock()
        _validate_aware_datetime(completed_at)
        report = _build_report(
            request=request,
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            overall_status="failed",
            exit_code=2,
            state=state,
            top_level_errors=(error,),
            manifest=None,
        )
        with _publication_lock:
            report_path, relative_report_path, report_sha256 = _publish_report_file(
                request.output_dir,
                started_at,
                run_id,
                report,
            )
            _publish_latest_report(
                request.output_dir,
                relative_report_path,
                "failed",
                2,
                None,
                report_sha256,
                completed_at,
            )
    except Exception:
        return _report_fallback_outcome(tracker)
    tracker.finish("failed")
    return SyncOutcome("failed", 2, report_path, None, (error,), state.counters)


def _report_fallback_outcome(tracker: _Progress) -> SyncOutcome:
    error = SyncError(
        "artifact",
        "artifact_write_failed",
        "A required synchronization report write failed.",
    )
    tracker.finish("failed")
    return SyncOutcome("failed", 2, None, None, (error,), SyncCounters())


def _build_report(
    *,
    request: SyncRequest,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    overall_status: OverallStatus,
    exit_code: int,
    state: _RunState,
    top_level_errors: tuple[SyncError, ...],
    manifest: PublishedManifest | None,
) -> dict[str, object]:
    config = state.config
    cache_result = state.cache_result
    inventory = state.inventory
    actions = cache_result.actions if cache_result is not None else ()
    miss_counts = {
        reason: sum(action.miss_reason == reason for action in actions)
        for reason in _CACHE_MISS_REASONS
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "run_id": run_id,
        "started_at": format_manifest_timestamp(started_at),
        "completed_at": format_manifest_timestamp(completed_at),
        "dry_run": request.dry_run,
        "overall_status": overall_status,
        "exit_code": exit_code,
        "source_endpoint_sha256": (config.source_endpoint_sha256 if config is not None else None),
        "source_bucket": config.bucket if config is not None else None,
        "cache_profile": CACHE_PROFILE,
        "filters": {
            "include_simfile_ids": sorted(request.include_simfile_ids),
            "exclude_simfile_ids": sorted(request.exclude_simfile_ids),
        },
        "network": {
            "head_concurrency": config.head_concurrency if config is not None else None,
            "download_concurrency": (config.download_concurrency if config is not None else None),
            "connect_timeout_seconds": (
                config.connect_timeout_seconds if config is not None else None
            ),
            "read_timeout_seconds": (config.read_timeout_seconds if config is not None else None),
            "retry_mode": "standard" if config is not None else None,
            "max_attempts": config.max_attempts if config is not None else None,
        },
        "artifacts": {
            "corpus_version": manifest.corpus_version if manifest is not None else None,
            "manifest_sha256": manifest.manifest_sha256 if manifest is not None else None,
            "manifest_path": manifest.relative_path if manifest is not None else None,
        },
        "counters": asdict(state.counters),
        "cache_misses_by_reason": miss_counts,
        "malformed_root_keys": (
            list(inventory.malformed_root_keys) if inventory is not None else []
        ),
        "ambiguous_prefixes": (
            {
                str(simfile_id): list(prefixes)
                for simfile_id, prefixes in sorted(inventory.ambiguous_prefixes.items())
            }
            if inventory is not None
            else {}
        ),
        "simfiles": _report_rows(cache_result),
        "errors": [_error_dict(error) for error in sorted(top_level_errors, key=_error_sort_key)],
    }


def _report_rows(cache_result: CacheSyncResult | None) -> list[dict[str, object]]:
    if cache_result is None:
        return []
    actions = {action.object_key: action for action in cache_result.actions}
    rows: list[dict[str, object]] = []
    for simfile in sorted(
        cache_result.simfiles,
        key=lambda item: (item.simfile_id, item.object_prefix),
    ):
        object_actions = []
        for remote in sorted(simfile.objects, key=lambda item: item.key):
            action = actions.get(remote.key)
            if action is None:
                continue
            object_actions.append(_action_dict(action))
        rows.append(
            {
                "simfile_id": simfile.simfile_id,
                "object_prefix": simfile.object_prefix,
                "sync_status": simfile.sync_status,
                "cache_actions": object_actions,
                "errors": [
                    _error_dict(error) for error in sorted(simfile.sync_errors, key=_error_sort_key)
                ],
            }
        )
    return rows


def _action_dict(action: CacheAction) -> dict[str, object]:
    return {
        "object_key": action.object_key,
        "action": action.action,
        "bytes": action.bytes,
        "miss_reason": action.miss_reason,
        "errors": [_error_dict(error) for error in sorted(action.errors, key=_error_sort_key)],
    }


def _build_counters(
    inventory: InventoryResult,
    cache_result: CacheSyncResult,
) -> SyncCounters:
    selected_sizes = {
        remote.key: remote.size
        for simfile in cache_result.simfiles
        for remote in simfile.objects
        if is_selected(remote.key)
    }
    attempted = tuple(action for action in cache_result.actions if action.miss_reason is not None)
    return SyncCounters(
        simfiles_discovered=inventory.simfiles_discovered,
        simfiles_included=len(cache_result.simfiles),
        simfiles_excluded_by_filter=inventory.simfiles_excluded_by_filter,
        simfiles_empty=sum(simfile.sync_status == "empty" for simfile in cache_result.simfiles),
        objects_listed=inventory.objects_listed,
        objects_selected=len(cache_result.actions),
        cache_hits=sum(action.action == "cache_hit" for action in cache_result.actions),
        downloads_planned=len(attempted),
        downloads_completed=sum(action.action == "downloaded" for action in cache_result.actions),
        downloads_failed=sum(action.action == "failed" for action in attempted),
        download_bytes_planned=sum(selected_sizes[action.object_key] for action in attempted),
        download_bytes_completed=sum(
            action.bytes for action in cache_result.actions if action.action == "downloaded"
        ),
    )


def _is_partial(
    inventory: InventoryResult,
    cache_result: CacheSyncResult,
) -> bool:
    return bool(
        inventory.root_errors
        or any(simfile.sync_status != "complete" for simfile in cache_result.simfiles)
        or any(action.action == "failed" for action in cache_result.actions)
    )


def _publish_report_file(
    output_dir: Path,
    started_at: datetime,
    run_id: str,
    report: dict[str, object],
) -> tuple[Path, str, str]:
    _ensure_durable_directory(output_dir)
    reports_dir = output_dir / "reports"
    _ensure_durable_directory(reports_dir)
    filename = report_filename(started_at, run_id)
    report_path = reports_dir / filename
    content = canonical_json_line(report)
    _atomic_replace_bytes(report_path, content)
    return report_path, f"reports/{filename}", sha256(content).hexdigest()


def _publish_latest_report(
    output_dir: Path,
    relative_report_path: str,
    overall_status: OverallStatus,
    exit_code: int,
    corpus_version: str | None,
    report_sha256: str,
    completed_at: datetime,
) -> None:
    payload = {
        "schema_version": REPORT_SCHEMA,
        "report_path": relative_report_path,
        "overall_status": overall_status,
        "exit_code": exit_code,
        "corpus_version": corpus_version,
        "report_sha256": report_sha256,
        "completed_at": format_manifest_timestamp(completed_at),
    }
    _atomic_replace_bytes(output_dir / "latest-report.json", canonical_json_line(payload))


def _snapshot_latest_manifest(output_dir: Path) -> bytes | None:
    path = output_dir / "latest.json"
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("manifest pointer is unavailable")
    return path.read_bytes()


def _restore_latest_manifest(output_dir: Path, previous_content: bytes | None) -> None:
    path = output_dir / "latest.json"
    if previous_content is not None:
        _atomic_replace_bytes(path, previous_content)
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(output_dir)


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    temporary_exists = False
    completed = False
    cleanup_failed = False
    try:
        with temporary_path.open("xb") as temporary:
            temporary_exists = True
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_exists = False
        _fsync_directory(path.parent)
        completed = True
    except Exception:
        completed = False
    finally:
        if temporary_exists:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                cleanup_failed = True
    if not completed or cleanup_failed:
        raise OSError("artifact publication failed")


def _ensure_durable_directory(path: Path) -> None:
    missing: list[Path] = []
    candidate = path
    while True:
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            missing.append(candidate)
            parent = candidate.parent
            if parent == candidate:
                raise OSError("artifact directory ancestor is unavailable")
            candidate = parent
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError("artifact directory path is unavailable")
        break
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)


def _fsync_directory(path: Path) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise OSError("directory durability support is unavailable")
    descriptor = os.open(path, os.O_RDONLY | no_follow | directory)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("artifact directory is unavailable")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _adapter_error(error: R2StoreError) -> SyncError:
    code: ErrorCode = error.code if error.code in _ADAPTER_CODES else "internal_error"
    object_key = error.object_key if error.object_key else None
    scope = "root" if code == "root_list_failed" else "configuration"
    if object_key is not None:
        scope = "object"
    return SyncError(scope, code, _SAFE_MESSAGES[code], object_key)


def _safe_error(code: ErrorCode) -> SyncError:
    if code in {"artifact_write_failed"}:
        scope = "artifact"
    elif code == "root_list_failed":
        scope = "root"
    else:
        scope = "configuration"
    return SyncError(scope, code, _SAFE_MESSAGES[code])


def _lock_error_code(error: RuntimeError) -> ErrorCode:
    if error.args == ("cache_locked",):
        return "cache_locked"
    if error.args == ("unsupported_platform",):
        return "unsupported_platform"
    return "artifact_write_failed"


def _error_dict(error: SyncError) -> dict[str, object]:
    return {
        "scope": error.scope,
        "code": error.code,
        "object_key": error.object_key,
        "message": error.message,
    }


def _error_sort_key(error: SyncError) -> tuple[str, str, str, str]:
    return error.scope, error.code, error.object_key or "", error.message


def _validate_request_paths(request: SyncRequest) -> None:
    for path in (request.output_dir, request.cache_dir):
        if not isinstance(path, Path):
            raise ValueError("artifact paths must be pathlib paths")
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ValueError("artifact path is unavailable")
    if request.provenance_file is not None and not isinstance(request.provenance_file, Path):
        raise ValueError("provenance path must be a pathlib path")


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str):
        raise ValueError("run ID must be a canonical UUID4")
    try:
        parsed = UUID(run_id)
    except (AttributeError, ValueError):
        raise ValueError("run ID must be a canonical UUID4") from None
    if parsed.version != 4 or str(parsed) != run_id:
        raise ValueError("run ID must be a canonical UUID4")


def _validate_aware_datetime(value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock values must be timezone-aware")
