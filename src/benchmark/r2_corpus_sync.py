from __future__ import annotations

import errno
import os
import stat
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
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
_OUTPUT_LOCK_FILENAME = ".r2-corpus-publication.lock"
_publication_lock = Lock()


class _SyncFailure(Exception):
    def __init__(self, error: SyncError):
        super().__init__(error.code)
        self.error = error


class _ReportWriteFailure(Exception):
    def __init__(self, primary_error: SyncError | None = None) -> None:
        super().__init__("report publication failed")
        self.primary_error = primary_error


class _InvalidPathFailure(Exception):
    pass


@dataclass
class _RunState:
    config: R2Config | None = None
    provenance: Mapping[int, ProvenanceRecord] | None = None
    inventory: InventoryResult | None = None
    cache_result: CacheSyncResult | None = None
    counters: SyncCounters = SyncCounters()
    immutable_manifest: PublishedManifest | None = None


@dataclass(frozen=True)
class _PointerSnapshots:
    latest_manifest: bytes | None
    latest_report: bytes | None


class _Progress:
    def __init__(self, callback: ProgressCallback, monotonic: MonotonicClock) -> None:
        self._callback = callback
        self._monotonic = monotonic
        self._entered: set[str] = set()
        self._last_item_event_at: dict[str, float] = {}
        self._lock = Lock()
        self._callback_enabled = True

    def _emit(self, event: ProgressEvent) -> None:
        if not self._callback_enabled:
            return
        try:
            self._callback(event)
        except Exception:
            self._callback_enabled = False

    def enter(self, phase: str, total: int | None = None) -> None:
        with self._lock:
            if phase in self._entered:
                return
            self._entered.add(phase)
            self._emit(ProgressEvent(phase, 0, total, f"{phase} phase started."))

    def begin_items(self, phase: str) -> None:
        with self._lock:
            self._last_item_event_at[phase] = self._monotonic()

    def item(
        self,
        phase: str,
        completed: int,
        total: int,
        completed_bytes: int,
        *,
        unit: str,
    ) -> None:
        with self._lock:
            now = self._monotonic()
            last_event_at = self._last_item_event_at.setdefault(phase, now)
            if completed % 100 == 0 or completed == total or now - last_event_at >= 5:
                self._emit(
                    ProgressEvent(
                        phase,
                        completed,
                        total,
                        (f"{phase}: {completed}/{total} {unit}, {completed_bytes} bytes."),
                    )
                )
                self._last_item_event_at[phase] = now

    def finish(self, status: OverallStatus) -> None:
        with self._lock:
            self._emit(ProgressEvent(status, 1, 1, f"{status} synchronization outcome."))


def utc_now() -> datetime:
    return _trusted_utc_now()


def new_run_id() -> str:
    return _trusted_run_id()


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
        _validate_aware_datetime(started_at)
        run_id = run_id_factory()
        _validate_run_id(run_id)
    except Exception:
        started_at = _trusted_utc_now()
        run_id = _trusted_run_id()
        tracker = _Progress(progress, monotonic)
        tracker.enter("configuration")
        return _fatal_outcome(
            request=request,
            run_id=run_id,
            started_at=started_at,
            clock=_trusted_utc_now,
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
        _validate_platform_support()
        dependency_check()
        state.config = R2Config.from_environ(environ)
        try:
            _validate_request_paths(request)
        except ValueError:
            raise _InvalidPathFailure from None
        try:
            state.provenance = load_provenance(request.provenance_file)
        except ValueError:
            raise _SyncFailure(_safe_error("provenance_invalid")) from None
    except R2StoreError as error:
        return _fatal_outcome(
            request, run_id, started_at, clock, tracker, state, _adapter_error(error)
        )
    except _SyncFailure as failure:
        if failure.error.code == "unsupported_platform":
            return _fatal_without_report(tracker, failure.error)
        return _fatal_outcome(request, run_id, started_at, clock, tracker, state, failure.error)
    except _InvalidPathFailure:
        return _fatal_without_report(tracker, _safe_error("invalid_config"))
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
            except _ReportWriteFailure as failure:
                return _report_fallback_outcome(tracker, failure.primary_error)
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
    tracker.begin_items("metadata")
    state.inventory = build_inventory(
        store,
        request.include_simfile_ids,
        request.exclude_simfile_ids,
        state.config.head_concurrency,
        item_progress=lambda completed, total, completed_bytes: tracker.item(
            "metadata",
            completed,
            total,
            completed_bytes,
            unit="objects",
        ),
    )

    tracker.enter("cache")
    tracker.begin_items("cache")
    state.cache_result = sync_cache(
        state.inventory.simfiles,
        store,
        index,
        state.config,
        request.dry_run,
        item_progress=lambda completed, total, completed_bytes: tracker.item(
            "cache",
            completed,
            total,
            completed_bytes,
            unit="objects",
        ),
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
    rendered_manifest: bytes | None = None
    if not request.dry_run:
        rows = build_manifest_rows(
            state.cache_result.simfiles,
            state.provenance,
            state.config.source_endpoint_sha256,
            state.config.bucket,
        )
        rendered_manifest = render_manifest(rows)

    tracker.enter("report")
    top_level_errors = state.inventory.root_errors
    with _output_publication_lock(request.output_dir):
        try:
            snapshots = _snapshot_pointers(request.output_dir)
        except Exception:
            raise _ReportWriteFailure from None

        if rendered_manifest is not None:
            try:
                state.immutable_manifest = publish_manifest(
                    request.output_dir,
                    rendered_manifest,
                )
            except ManifestPublicationError as error:
                _restore_pointers_or_report_failure(
                    request.output_dir,
                    snapshots,
                    error.error,
                )
                raise
            except Exception:
                _restore_pointers_or_report_failure(request.output_dir, snapshots)
                raise _ReportWriteFailure from None

        try:
            completed_at = clock()
            _validate_aware_datetime(completed_at)
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
        except Exception:
            _restore_pointers_or_report_failure(
                request.output_dir,
                snapshots,
                _safe_error("internal_error"),
            )
            raise

        report_path: Path | None = None
        try:
            report_path, relative_report_path, report_sha256 = _publish_report_file(
                request.output_dir,
                started_at,
                run_id,
                report,
            )
            if state.immutable_manifest is not None:
                publish_latest_manifest(
                    request.output_dir,
                    state.immutable_manifest,
                    overall_status,
                    completed_at,
                )
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
        except Exception as error:
            publication_error = _publication_error(error)
            try:
                _restore_pointers(request.output_dir, snapshots)
            except Exception:
                _remove_attempt_report(request.output_dir, started_at, run_id)
                raise _ReportWriteFailure(publication_error) from None
            return _rewrite_attempt_as_failed(
                request=request,
                run_id=run_id,
                started_at=started_at,
                completed_at=completed_at,
                tracker=tracker,
                state=state,
                error=publication_error,
            )

    tracker.finish(overall_status)
    return SyncOutcome(
        overall_status=overall_status,
        exit_code=exit_code,
        report_path=report_path,
        manifest=state.immutable_manifest,
        errors=top_level_errors,
        counters=state.counters,
    )


def _rewrite_attempt_as_failed(
    *,
    request: SyncRequest,
    run_id: str,
    started_at: datetime,
    completed_at: datetime,
    tracker: _Progress,
    state: _RunState,
    error: SyncError,
) -> SyncOutcome:
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
    try:
        report_path, _, _ = _publish_report_file(
            request.output_dir,
            started_at,
            run_id,
            report,
        )
    except Exception:
        _remove_attempt_report(request.output_dir, started_at, run_id)
        raise _ReportWriteFailure(error) from None
    tracker.finish("failed")
    return SyncOutcome("failed", 2, report_path, None, (error,), state.counters)


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
        completed_at = _safe_clock_value(clock)
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
        with _output_publication_lock(request.output_dir):
            snapshots = _snapshot_pointers(request.output_dir)
            try:
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
                _restore_pointers_or_report_failure(request.output_dir, snapshots)
                raise
    except Exception:
        return _report_fallback_outcome(tracker, error)
    tracker.finish("failed")
    return SyncOutcome("failed", 2, report_path, None, (error,), state.counters)


def _report_fallback_outcome(
    tracker: _Progress,
    primary_error: SyncError | None = None,
) -> SyncOutcome:
    artifact_error = SyncError(
        "artifact",
        "artifact_write_failed",
        "A required synchronization report write failed.",
    )
    tracker.finish("failed")
    if primary_error is None or primary_error.code == artifact_error.code:
        errors = (artifact_error,)
    else:
        errors = (primary_error, artifact_error)
    return SyncOutcome("failed", 2, None, None, errors, SyncCounters())


def _fatal_without_report(tracker: _Progress, error: SyncError) -> SyncOutcome:
    tracker.finish("failed")
    return SyncOutcome("failed", 2, None, None, (error,), SyncCounters())


def _publication_error(error: Exception) -> SyncError:
    if isinstance(error, ManifestPublicationError):
        return error.error
    return _safe_error("artifact_write_failed")


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


@contextmanager
def _output_publication_lock(output_dir: Path):
    with _publication_lock:
        _ensure_durable_directory(output_dir)
        descriptor, flock, unlock_operation = _acquire_output_lock(output_dir)
        try:
            yield
        finally:
            try:
                flock(descriptor, unlock_operation)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass


def _load_fcntl():
    import fcntl

    return fcntl


def _validate_platform_support() -> None:
    if getattr(os, "O_NOFOLLOW", None) is None or getattr(os, "O_DIRECTORY", None) is None:
        raise _SyncFailure(_safe_error("unsupported_platform"))
    try:
        fcntl = _load_fcntl()
    except (ImportError, AttributeError):
        raise _SyncFailure(_safe_error("unsupported_platform")) from None
    if (
        not callable(getattr(fcntl, "flock", None))
        or getattr(fcntl, "LOCK_EX", None) is None
        or getattr(fcntl, "LOCK_NB", None) is None
        or getattr(fcntl, "LOCK_UN", None) is None
    ):
        raise _SyncFailure(_safe_error("unsupported_platform"))


def _acquire_output_lock(output_dir: Path):
    _validate_platform_support()
    no_follow = os.O_NOFOLLOW
    fcntl = _load_fcntl()

    lock_path = output_dir / _OUTPUT_LOCK_FILENAME
    descriptor = -1
    try:
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | no_follow,
            0o600,
        )
        descriptor_metadata = os.fstat(descriptor)
        path_metadata = lock_path.lstat()
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or not stat.S_ISREG(path_metadata.st_mode)
            or descriptor_metadata.st_dev != path_metadata.st_dev
            or descriptor_metadata.st_ino != path_metadata.st_ino
        ):
            raise OSError("publication lock is unavailable")
        _fsync_directory(output_dir)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as error:
            unsupported = {
                getattr(errno, "ENOSYS", -1),
                getattr(errno, "ENOTSUP", -1),
                getattr(errno, "EOPNOTSUPP", -1),
            }
            if error.errno in unsupported:
                raise _SyncFailure(_safe_error("unsupported_platform")) from None
            raise
    except _SyncFailure:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        raise _SyncFailure(_safe_error("artifact_write_failed")) from None
    return descriptor, fcntl.flock, fcntl.LOCK_UN


def _snapshot_pointers(output_dir: Path) -> _PointerSnapshots:
    return _PointerSnapshots(
        latest_manifest=_snapshot_pointer(output_dir / "latest.json"),
        latest_report=_snapshot_pointer(output_dir / "latest-report.json"),
    )


def _snapshot_pointer(path: Path) -> bytes | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None or not stat.S_ISREG(metadata.st_mode):
        raise OSError("artifact pointer is unavailable")
    descriptor = os.open(path, os.O_RDONLY | no_follow)
    try:
        descriptor_metadata = os.fstat(descriptor)
        current_metadata = path.lstat()
        if (
            not stat.S_ISREG(descriptor_metadata.st_mode)
            or not stat.S_ISREG(current_metadata.st_mode)
            or descriptor_metadata.st_dev != current_metadata.st_dev
            or descriptor_metadata.st_ino != current_metadata.st_ino
        ):
            raise OSError("artifact pointer is unavailable")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _restore_pointers(output_dir: Path, snapshots: _PointerSnapshots) -> None:
    failures: list[Exception] = []
    for path, content in (
        (output_dir / "latest.json", snapshots.latest_manifest),
        (output_dir / "latest-report.json", snapshots.latest_report),
    ):
        try:
            _restore_pointer(path, content)
        except Exception as error:
            failures.append(error)
    if failures:
        raise OSError("artifact pointer rollback failed")


def _restore_pointers_or_report_failure(
    output_dir: Path,
    snapshots: _PointerSnapshots,
    primary_error: SyncError | None = None,
) -> None:
    try:
        _restore_pointers(output_dir, snapshots)
    except Exception:
        raise _ReportWriteFailure(primary_error) from None


def _restore_pointer(path: Path, previous_content: bytes | None) -> None:
    if previous_content is not None:
        _atomic_replace_bytes(path, previous_content)
        return
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("artifact pointer is unavailable")
    path.unlink()
    _fsync_directory(path.parent)


def _remove_attempt_report(output_dir: Path, started_at: datetime, run_id: str) -> None:
    path = output_dir / "reports" / report_filename(started_at, run_id)
    try:
        metadata = path.lstat()
    except (FileNotFoundError, OSError):
        return
    if not stat.S_ISREG(metadata.st_mode):
        return
    try:
        path.unlink()
        _fsync_directory(path.parent)
    except OSError:
        return


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
        try:
            directory.mkdir()
        except FileExistsError:
            metadata = directory.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise OSError("artifact directory path is unavailable") from None
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
        _validate_directory_components(path)
    if request.provenance_file is not None and not isinstance(request.provenance_file, Path):
        raise ValueError("provenance path must be a pathlib path")


def _validate_directory_components(path: Path) -> None:
    if ".." in path.parts:
        raise ValueError("artifact path may not contain parent traversal")
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return
        except OSError:
            raise ValueError("artifact path is unavailable") from None
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("artifact path is unavailable")


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


def _safe_clock_value(clock: Clock) -> datetime:
    try:
        value = clock()
        _validate_aware_datetime(value)
    except Exception:
        return _trusted_utc_now()
    return value


def _trusted_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trusted_run_id() -> str:
    return str(uuid4())
