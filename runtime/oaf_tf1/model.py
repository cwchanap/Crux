"""Dependency-light OaF model and checkpoint configuration."""

# TensorFlow and vendored Magenta remain lazy so host-side imports are cheap.
# pylint: disable=import-error,import-outside-toplevel

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

_SCHEMA = "crux.oaf-model/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EXPECTED_COMPONENT_NAMES = (
    "model.ckpt-569400.data-00000-of-00001",
    "model.ckpt-569400.index",
    "model.ckpt-569400.meta",
)
_MODEL_KEYS = frozenset(
    {
        "schema",
        "backend_id",
        "model_id",
        "architecture_id",
        "upstream_source_commit",
        "training_data_map_id",
        "native_output_space_id",
        "native_metadata_schema_id",
        "max_input_audio_frames",
        "checkpoint",
    }
)
_CHECKPOINT_KEYS = frozenset({"url", "archive_name", "archive_sha256", "components"})


class OafModelConfigError(ValueError):
    """The OaF model configuration is malformed or unsafe."""


@dataclass(frozen=True)
class OafCheckpointConfig:
    url: str
    archive_name: str
    archive_sha256: str
    components: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url:
            raise OafModelConfigError("checkpoint.url is invalid")
        if (
            not isinstance(self.archive_name, str)
            or not self.archive_name
            or self.archive_name in {".", ".."}
            or any(separator in self.archive_name for separator in ("/", "\\", ":", "\x00"))
        ):
            raise OafModelConfigError("checkpoint.archive_name is invalid")
        _require_sha256(self.archive_sha256, "checkpoint.archive_sha256")
        if not isinstance(self.components, Mapping):
            raise OafModelConfigError("checkpoint.components is invalid")
        components = dict(self.components)
        if set(components) != set(_EXPECTED_COMPONENT_NAMES):
            raise OafModelConfigError("checkpoint.components names are invalid")
        for name in _EXPECTED_COMPONENT_NAMES:
            _require_sha256(components[name], f"checkpoint.components.{name}")
        object.__setattr__(self, "components", MappingProxyType(components))


@dataclass(frozen=True)
class OafModelConfig:
    backend_id: str
    model_id: str
    architecture_id: str
    upstream_source_commit: str
    training_data_map_id: str
    native_output_space_id: str
    native_metadata_schema_id: str
    max_input_audio_frames: int | None
    checkpoint: OafCheckpointConfig

    def __post_init__(self) -> None:
        for field in (
            "backend_id",
            "model_id",
            "architecture_id",
            "training_data_map_id",
            "native_output_space_id",
            "native_metadata_schema_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise OafModelConfigError(f"{field} is invalid")
        if not isinstance(self.upstream_source_commit, str) or not _COMMIT.fullmatch(
            self.upstream_source_commit
        ):
            raise OafModelConfigError("upstream_source_commit is invalid")
        if self.max_input_audio_frames is not None and (
            not isinstance(self.max_input_audio_frames, int)
            or isinstance(self.max_input_audio_frames, bool)
            or self.max_input_audio_frames < 0
        ):
            raise OafModelConfigError("max_input_audio_frames is invalid")
        if not isinstance(self.checkpoint, OafCheckpointConfig):
            raise OafModelConfigError("checkpoint is invalid")


def load_model_config(path: Path = Path("runtime/oaf_tf1/model.json")) -> OafModelConfig:
    """Load and validate one repository-authored OaF model configuration."""
    try:
        content = _read_regular_file_no_follow(path)
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        raise OafModelConfigError("model configuration is invalid") from None
    if not isinstance(payload, dict) or set(payload) != _MODEL_KEYS:
        raise OafModelConfigError("model configuration keys are invalid")
    if payload.get("schema") != _SCHEMA:
        raise OafModelConfigError("model configuration schema is invalid")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or set(checkpoint) != _CHECKPOINT_KEYS:
        raise OafModelConfigError("checkpoint keys are invalid")
    try:
        return OafModelConfig(
            backend_id=payload["backend_id"],
            model_id=payload["model_id"],
            architecture_id=payload["architecture_id"],
            upstream_source_commit=payload["upstream_source_commit"],
            training_data_map_id=payload["training_data_map_id"],
            native_output_space_id=payload["native_output_space_id"],
            native_metadata_schema_id=payload["native_metadata_schema_id"],
            max_input_audio_frames=payload["max_input_audio_frames"],
            checkpoint=OafCheckpointConfig(
                url=checkpoint["url"],
                archive_name=checkpoint["archive_name"],
                archive_sha256=checkpoint["archive_sha256"],
                components=checkpoint["components"],
            ),
        )
    except (KeyError, TypeError, OafModelConfigError) as error:
        if isinstance(error, OafModelConfigError):
            raise
        raise OafModelConfigError("model configuration fields are invalid") from None


def _require_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise OafModelConfigError(f"{field} is invalid")


def _read_regular_file_no_follow(path: Path) -> bytes:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, os.O_RDONLY | no_follow)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("model configuration is not a regular file")
        return os.read(descriptor, metadata.st_size)
    finally:
        os.close(descriptor)


# The model engine intentionally owns only inference concerns.  Lock, seal,
# attestation, and publication code stays in the retained adapter until the
# atomic cutover task.
CHECKPOINT_COUNT = 130
REQUIRED_INFERENCE_COUNT = 78
NON_INFERENCE_COUNT = 52
MIN_MIDI_PITCH = 21
MODEL_SCRATCH_DIRECTORY = "/tmp/crux-oaf-model"
DEFAULT_UNINSTRUMENTED_SEQUENCE_LIB = Path("/opt/crux/upstream/magenta/music/sequences_lib.py")
_STOCHASTIC_OPERATION_TYPES = frozenset(
    {
        "Multinomial",
        "ParameterizedTruncatedNormal",
        "RandomGamma",
        "RandomPoisson",
        "RandomShuffle",
        "RandomStandardNormal",
        "RandomUniform",
        "RandomUniformInt",
        "StatelessMultinomial",
        "StatelessRandomGetKeyCounter",
        "StatelessRandomNormal",
        "StatelessRandomUniform",
        "StatelessRandomUniformInt",
        "StatelessTruncatedNormal",
        "TruncatedNormal",
    }
)
_CALIBRATION_TRAINING_GROUPS = (
    {"base_midi": 36, "group_id": "kick", "member_pitches": [36], "output_bin": 15},
    {
        "base_midi": 38,
        "group_id": "snare",
        "member_pitches": [38, 40, 37, 39],
        "output_bin": 17,
    },
    {
        "base_midi": 48,
        "group_id": "toms",
        "member_pitches": [48, 50, 45, 47, 43, 58, 64],
        "output_bin": 27,
    },
    {
        "base_midi": 46,
        "group_id": "hihat",
        "member_pitches": [46, 26, 42, 22, 44, 54, 70],
        "output_bin": 25,
    },
    {"base_midi": 51, "group_id": "ride", "member_pitches": [51, 59], "output_bin": 30},
    {
        "base_midi": 53,
        "group_id": "ride_bell",
        "member_pitches": [53, 56],
        "output_bin": 32,
    },
    {
        "base_midi": 49,
        "group_id": "crash",
        "member_pitches": [49, 55, 57, 52],
        "output_bin": 28,
    },
    {"base_midi": 75, "group_id": "sticks", "member_pitches": [75], "output_bin": 54},
)
CALIBRATION_TRAINING_GROUPS = _CALIBRATION_TRAINING_GROUPS


class OafModelError(RuntimeError):
    """The extracted OaF model failed an inference or integrity check."""


@dataclass(frozen=True)
class OafNativeEvent:
    time_sec: float
    native_class_id: str
    model_output_bin: int
    native_midi_note: int
    upstream_8hit_group_id: str | None
    confidence: float
    velocity_midi: int


@dataclass(frozen=True)
class TensorCoverage:
    checkpoint_count: int
    required_count: int
    restored_count: int
    non_inference_count: int
    checkpoint_inventory_sha256: str
    required_inventory_sha256: str
    non_inference_inventory_sha256: str


def frame_time_seconds(frame_index: int) -> float:
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise OafModelError("emitted frame index is invalid")
    return frame_index * 512 / 44100


def velocity_to_midi(raw_velocity: float) -> int:
    value = float(raw_velocity)
    if not math.isfinite(value):
        raise OafModelError("emitted raw velocity is not finite")
    return int(max(min(value, 1.0), 0.0) * 127)


def _group_for_pitch(
    native_midi_note: int,
    training_groups: Sequence[Mapping[str, Any]] = CALIBRATION_TRAINING_GROUPS,
) -> str | None:
    matches: list[str] = []
    for group in training_groups:
        members = group.get("member_pitches")
        if isinstance(members, (list, tuple)) and native_midi_note in members:
            group_id = group.get("group_id")
            if not isinstance(group_id, str) or not group_id:
                raise OafModelError("locked training group is invalid")
            matches.append(group_id)
    if len(matches) > 1:
        raise OafModelError("emitted pitch has ambiguous training groups")
    return matches[0] if matches else None


def native_event_from_capture(
    *,
    start_frame: int,
    native_midi_note: int,
    raw_velocity: float,
    raw_confidence: float,
    training_groups: Sequence[Mapping[str, Any]] = CALIBRATION_TRAINING_GROUPS,
) -> OafNativeEvent:
    if (
        isinstance(native_midi_note, bool)
        or not isinstance(native_midi_note, int)
        or not MIN_MIDI_PITCH <= native_midi_note <= 108
    ):
        raise OafModelError("emitted native MIDI pitch is outside the output space")
    confidence = float(raw_confidence)
    if not math.isfinite(confidence):
        raise OafModelError("emitted raw confidence is not finite")
    return OafNativeEvent(
        time_sec=frame_time_seconds(start_frame),
        native_class_id="midi_" + str(native_midi_note),
        model_output_bin=native_midi_note - MIN_MIDI_PITCH,
        native_midi_note=native_midi_note,
        upstream_8hit_group_id=_group_for_pitch(native_midi_note, training_groups),
        confidence=confidence,
        velocity_midi=velocity_to_midi(raw_velocity),
    )


# A short alias keeps pure conversion call sites readable while preserving the
# extracted helper's descriptive name for parity with the old adapter.
event_from_capture = native_event_from_capture


def _normalize_inventory(
    values: Sequence[Mapping[str, Any]], *, include_reason: bool, label: str
) -> tuple[Mapping[str, Any], ...]:
    normalized: list[Mapping[str, Any]] = []
    names: set[str] = set()
    expected_keys = {"dtype", "name", "shape"}
    if include_reason:
        expected_keys.add("reason")
    for value in values:
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise OafModelError(label + " entry does not match the exact schema")
        name = value.get("name")
        dtype = value.get("dtype")
        shape = value.get("shape")
        if not isinstance(name, str) or not name or name in names:
            raise OafModelError(label + " names are invalid or duplicate")
        if not isinstance(dtype, str) or not dtype:
            raise OafModelError(label + " dtype is invalid")
        if not isinstance(shape, (list, tuple)) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        ):
            raise OafModelError(label + " shape is invalid")
        entry: dict[str, Any] = {"dtype": dtype, "name": name, "shape": list(shape)}
        if include_reason:
            reason = value.get("reason")
            if not isinstance(reason, str) or not reason:
                raise OafModelError(label + " reason is invalid")
            entry["reason"] = reason
        normalized.append(entry)
        names.add(name)
    if [entry["name"] for entry in normalized] != sorted(
        names, key=lambda item: item.encode("utf-8")
    ):
        raise OafModelError(label + " entries are not bytewise sorted")
    return tuple(normalized)


def _inventory_sha256(values: Sequence[Mapping[str, Any]]) -> str:
    content = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def validate_restored_tensor_inventory(
    restored_names: Sequence[str], *, required_names: Sequence[str] | None = None
) -> int:
    """Require every graph tensor to be present in the restored checkpoint."""
    restored = tuple(restored_names)
    if any(not isinstance(name, str) or not name for name in restored):
        raise OafModelError("restored tensor names are invalid")
    if len(set(restored)) != len(restored):
        raise OafModelError("restored tensor names are duplicate")
    if required_names is None:
        if len(restored) != REQUIRED_INFERENCE_COUNT:
            raise OafModelError(
                f"required inference tensor count is {len(restored)}, expected "
                f"{REQUIRED_INFERENCE_COUNT}"
            )
        return len(restored)
    required = tuple(required_names)
    missing = sorted(set(required) - set(restored), key=lambda name: name.encode("utf-8"))
    if missing:
        raise OafModelError(f"required inference tensor missing: {missing[0]}")
    if len(required) != REQUIRED_INFERENCE_COUNT:
        raise OafModelError(
            f"required inference tensor count is {len(required)}, expected "
            f"{REQUIRED_INFERENCE_COUNT}"
        )
    return len(required)


validate_required_tensors = validate_restored_tensor_inventory


def validate_tensor_coverage(
    *,
    checkpoint_inventory: Sequence[Mapping[str, Any]],
    required_inventory: Sequence[Mapping[str, Any]],
    non_inference_inventory: Sequence[Mapping[str, Any]],
    graph_inventory: Sequence[Mapping[str, Any]],
    uninitialized_required: Sequence[str],
) -> TensorCoverage:
    """Validate the exact released 130-tensor checkpoint partition."""
    if len(checkpoint_inventory) != CHECKPOINT_COUNT:
        raise OafModelError("checkpoint count must be exactly 130")
    if len(required_inventory) != REQUIRED_INFERENCE_COUNT:
        raise OafModelError("required count must be exactly 78")
    if len(non_inference_inventory) != NON_INFERENCE_COUNT:
        raise OafModelError("non-inference count must be exactly 52")
    checkpoint = _normalize_inventory(
        checkpoint_inventory, include_reason=False, label="checkpoint inventory"
    )
    required = _normalize_inventory(
        required_inventory, include_reason=False, label="required inventory"
    )
    non_inference = _normalize_inventory(
        non_inference_inventory, include_reason=True, label="non-inference inventory"
    )
    graph = _normalize_inventory(graph_inventory, include_reason=False, label="graph inventory")
    required_names = {entry["name"] for entry in required}
    checkpoint_names = {entry["name"] for entry in checkpoint}
    missing = sorted(required_names - checkpoint_names, key=lambda name: name.encode("utf-8"))
    if missing:
        raise OafModelError(f"required inference tensor missing: {missing[0]}")
    if graph != required:
        raise OafModelError("graph inventory does not match required inventory")
    classified_non_inference = tuple(
        {key: value for key, value in entry.items() if key != "reason"} for entry in non_inference
    )
    classified = tuple(
        sorted(required + classified_non_inference, key=lambda entry: entry["name"].encode("utf-8"))
    )
    if checkpoint != classified:
        raise OafModelError(
            "checkpoint inventory does not match the locked required/non-inference partition"
        )
    if uninitialized_required:
        missing_name = sorted(uninitialized_required, key=lambda name: name.encode("utf-8"))[0]
        raise OafModelError(f"required inference tensor remains uninitialized: {missing_name}")
    return TensorCoverage(
        checkpoint_count=len(checkpoint),
        required_count=len(required),
        restored_count=len(required),
        non_inference_count=len(non_inference),
        checkpoint_inventory_sha256=_inventory_sha256(checkpoint),
        required_inventory_sha256=_inventory_sha256(required),
        non_inference_inventory_sha256=_inventory_sha256(non_inference),
    )


def assert_no_reachable_stochastic_ops(fetch_operations: Iterable[Any]) -> None:
    """Reject random operations reachable from the prediction fetches."""
    pending = list(fetch_operations)
    visited: set[int] = set()
    while pending:
        operation = pending.pop()
        if getattr(operation, "type", None) is None:
            operation = getattr(operation, "op", operation)
        identity = id(operation)
        if identity in visited:
            continue
        visited.add(identity)
        if getattr(operation, "type", None) in _STOCHASTIC_OPERATION_TYPES:
            raise OafModelError("PREDICT output has a reachable stochastic TensorFlow operation")
        for tensor in getattr(operation, "inputs", ()):
            input_operation = getattr(tensor, "op", None)
            if input_operation is not None:
                pending.append(input_operation)
        pending.extend(getattr(operation, "control_inputs", ()))


def _flatten_prediction_operations(value: Any) -> tuple[Any, ...]:
    operations: list[Any] = []
    if isinstance(value, Mapping):
        for item in value.values():
            operations.extend(_flatten_prediction_operations(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            operations.extend(_flatten_prediction_operations(item))
    else:
        operation = getattr(value, "op", None)
        if operation is not None:
            operations.append(operation)
    return tuple(operations)


def _build_coverage_graph(tf: Any, config: Any, hparams: Any):
    from magenta.models.onsets_frames_transcription import data

    graph = tf.Graph()
    with graph.as_default():
        examples = tf.placeholder(tf.string, [None], name="canonical_examples")
        dataset = data.provide_batch(
            examples=examples,
            preprocess_examples=True,
            params=hparams,
            is_training=False,
            shuffle_examples=False,
            skip_n_initial_records=0,
        )
        iterator = dataset.make_initializable_iterator()
        features, labels = iterator.get_next()
        estimator_spec = config.model_fn(
            features, labels, tf.estimator.ModeKeys.PREDICT, hparams, None
        )
        variables = tuple(tf.global_variables())
        predictions = estimator_spec.predictions
    return graph, variables, predictions


def _tensorflow_graph_inventory(variables: Sequence[Any]) -> tuple[Mapping[str, Any], ...]:
    values = []
    for variable in variables:
        values.append(
            {
                "dtype": variable.dtype.base_dtype.name,
                "name": variable.op.name,
                "shape": variable.shape.as_list(),
            }
        )
    return tuple(sorted(values, key=lambda entry: entry["name"].encode("utf-8")))


def _tensorflow_checkpoint_inventory(tf: Any, checkpoint_prefix: str):
    reader = tf.train.NewCheckpointReader(checkpoint_prefix)
    dtype_map = reader.get_variable_to_dtype_map()
    inventory = []
    for name, shape in tf.train.list_variables(checkpoint_prefix):
        dtype = dtype_map[name]
        inventory.append(
            {"dtype": getattr(dtype, "name", str(dtype)), "name": name, "shape": list(shape)}
        )
    return tuple(sorted(inventory, key=lambda entry: entry["name"].encode("utf-8")))


def _checkpoint_prefix(checkpoint_dir: Path, config: OafModelConfig) -> str:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise OafModelError("checkpoint directory is unavailable")
    names = config.checkpoint.components
    missing = [name for name in names if not (checkpoint_dir / name).is_file()]
    if missing:
        raise OafModelError(f"checkpoint component is missing: {missing[0]}")
    # Components are fixed by model.json and all share this prefix.
    return str(checkpoint_dir / "model.ckpt-569400")


def _plain_inventory(values: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {"dtype": str(value["dtype"]), "name": str(value["name"]), "shape": list(value["shape"])}
        for value in values
    )


def _load_uninstrumented_sequences_module(source_path: Path | None = None) -> Any:
    source = source_path or DEFAULT_UNINSTRUMENTED_SEQUENCE_LIB
    try:
        status = source.lstat()
        if not stat.S_ISREG(status.st_mode):
            raise OSError("unmodified converter is not a regular file")
        spec = importlib.util.spec_from_file_location("_crux_uninstrumented_sequences_lib", source)
        if spec is None or spec.loader is None:
            raise OSError("unmodified converter has no source loader")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (OSError, ImportError, ValueError) as error:
        raise OafModelError("unmodified sequence converter is unavailable") from error


def _serialized_example(audio_path: Path) -> bytes:
    from magenta.models.onsets_frames_transcription import audio_label_data_utils
    from magenta.music.protobuf import music_pb2

    try:
        content = Path(audio_path).read_bytes()
    except OSError as error:
        raise OafModelError("audio input is unavailable") from error
    examples = list(
        audio_label_data_utils.process_record(
            wav_data=content,
            sample_rate=44100,
            ns=music_pb2.NoteSequence(),
            example_id=Path(audio_path).name,
            min_length=0,
            max_length=-1,
            allow_empty_notesequence=True,
            load_audio_with_librosa=False,
        )
    )
    if len(examples) != 1:
        raise OafModelError("canonical preprocessing did not return one example")
    return examples[0].SerializeToString()


@dataclass(frozen=True)
class _LoadedModelState:
    estimator: Any
    hparams: Any
    checkpoint_prefix: str
    coverage: TensorCoverage
    training_groups: Sequence[Mapping[str, Any]]
    uninstrumented_sequences_module: Any


class OafModel:
    """TensorFlow 1 OaF model engine extracted from the sealed adapter."""

    def __init__(self, state: _LoadedModelState, config: OafModelConfig) -> None:
        self._state = state
        self.config = config

    @classmethod
    def load(cls, checkpoint_dir: Path, config: OafModelConfig | None = None) -> "OafModel":
        config = config or load_model_config()
        checkpoint_prefix = _checkpoint_prefix(Path(checkpoint_dir), config)
        try:
            import tensorflow.compat.v1 as tf
            from magenta.models.onsets_frames_transcription import configs, train_util
        except (ImportError, ModuleNotFoundError) as error:
            raise OafModelError("TensorFlow OaF runtime is unavailable") from error

        model_config = configs.CONFIG_MAP["drums"]
        hparams = copy.deepcopy(model_config.hparams)
        hparams.batch_size = 1
        hparams.truncated_length_secs = 0
        graph, variables, predictions = _build_coverage_graph(tf, model_config, hparams)
        graph_inventory = _tensorflow_graph_inventory(variables)
        checkpoint_inventory = _tensorflow_checkpoint_inventory(tf, checkpoint_prefix)
        required_inventory = _plain_inventory(graph_inventory)
        required_names = {entry["name"] for entry in required_inventory}
        checkpoint_names = {entry["name"] for entry in checkpoint_inventory}
        missing = sorted(required_names - checkpoint_names, key=lambda name: name.encode("utf-8"))
        if missing:
            raise OafModelError(f"required inference tensor missing: {missing[0]}")
        if len(required_inventory) != REQUIRED_INFERENCE_COUNT:
            raise OafModelError(
                f"required inference tensor count is {len(required_inventory)}, expected "
                f"{REQUIRED_INFERENCE_COUNT}"
            )
        non_inference_inventory = tuple(
            {**entry, "reason": "not_in_prediction_graph"}
            for entry in checkpoint_inventory
            if entry["name"] not in required_names
        )
        required_variables = {
            variable.op.name: variable
            for variable in variables
            if variable.op.name in required_names
        }
        with graph.as_default():
            required_saver = tf.train.Saver(var_list=required_variables)
            required_uninitialized = tf.report_uninitialized_variables(
                var_list=list(required_variables.values())
            )
        with tf.Session(graph=graph) as session:
            try:
                required_saver.restore(session, checkpoint_prefix)
                uninitialized_raw = session.run(required_uninitialized)
                uninitialized = tuple(
                    (
                        value.decode("utf-8", errors="strict")
                        if isinstance(value, bytes)
                        else str(value)
                    )
                    for value in uninitialized_raw
                )
            except Exception as error:
                raise OafModelError("checkpoint restore failed") from error
            assert_no_reachable_stochastic_ops(_flatten_prediction_operations(predictions))
        coverage = validate_tensor_coverage(
            checkpoint_inventory=checkpoint_inventory,
            required_inventory=required_inventory,
            non_inference_inventory=non_inference_inventory,
            graph_inventory=graph_inventory,
            uninitialized_required=uninitialized,
        )
        estimator = train_util.create_estimator(
            model_config.model_fn, MODEL_SCRATCH_DIRECTORY, hparams
        )
        state = _LoadedModelState(
            estimator=estimator,
            hparams=hparams,
            checkpoint_prefix=checkpoint_prefix,
            coverage=coverage,
            training_groups=CALIBRATION_TRAINING_GROUPS,
            uninstrumented_sequences_module=_load_uninstrumented_sequences_module(),
        )
        return cls(state, config)

    @property
    def restored_tensor_count(self) -> int:
        return self._state.coverage.restored_count

    def transcribe(self, audio_path: Path) -> tuple[OafNativeEvent, ...]:
        try:
            import tensorflow.compat.v1 as tf
            from magenta.models.onsets_frames_transcription import data, infer_util

            serialized = _serialized_example(Path(audio_path))

            def transcription_data(params):
                return data.provide_batch(
                    examples=tf.constant([serialized], dtype=tf.string),
                    preprocess_examples=True,
                    params=params,
                    is_training=False,
                    shuffle_examples=False,
                    skip_n_initial_records=0,
                )

            input_fn = infer_util.labels_to_features_wrapper(transcription_data)
            predictions = list(
                self._state.estimator.predict(
                    input_fn,
                    checkpoint_path=self._state.checkpoint_prefix,
                    yield_single_examples=False,
                )
            )
            if len(predictions) != 1:
                raise OafModelError("frozen inference did not return exactly one prediction batch")
            prediction = predictions[0]
            sequence, captured = infer_util.predict_sequence(
                frame_probs=prediction["frame_probs"][0],
                onset_probs=prediction["onset_probs"][0],
                frame_predictions=prediction["frame_predictions"][0],
                onset_predictions=prediction["onset_predictions"][0],
                offset_predictions=prediction["offset_predictions"][0],
                velocity_values=prediction["velocity_values"][0],
                min_pitch=MIN_MIDI_PITCH,
                hparams=self._state.hparams,
                onsets_only=True,
                capture_emitted_frames=True,
            )
            upstream_sequence = (
                self._state.uninstrumented_sequences_module.pianoroll_onsets_to_note_sequence(
                    onsets=prediction["onset_predictions"][0],
                    frames_per_second=data.hparams_frames_per_second(self._state.hparams),
                    note_duration_seconds=0.05,
                    min_midi_pitch=MIN_MIDI_PITCH,
                    velocity_values=prediction["velocity_values"][0],
                    velocity_scale=self._state.hparams.velocity_scale,
                    velocity_bias=self._state.hparams.velocity_bias,
                )
            )
            if sequence.SerializeToString() != upstream_sequence.SerializeToString():
                raise OafModelError("instrumented sequence differs from upstream prediction")
            paired = infer_util.pair_emitted_frame_confidence(
                captured, prediction["onset_probs"][0], min_pitch=MIN_MIDI_PITCH
            )
            return tuple(
                native_event_from_capture(
                    start_frame=int(start_frame),
                    native_midi_note=int(native_midi_note),
                    raw_velocity=float(raw_velocity),
                    raw_confidence=float(raw_confidence),
                    training_groups=self._state.training_groups,
                )
                for start_frame, native_midi_note, raw_velocity, raw_confidence in paired
            )
        except OafModelError:
            raise
        except (SystemExit, KeyboardInterrupt, MemoryError):
            raise
        except Exception as error:
            raise OafModelError("inference failed") from error
