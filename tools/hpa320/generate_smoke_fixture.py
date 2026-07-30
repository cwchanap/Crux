#!/usr/bin/env python3
"""Generate the canonical integer-only HPA-320 smoke WAV."""

# Exact-type checks intentionally reject bools from the canonical numeric schema.
# pylint: disable=unidiomatic-typecheck
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.benchmark.backend_identity import canonical_json_bytes, strict_json_loads
from src.benchmark.backend_publication import atomic_replace_bytes
from src.benchmark.input_view import parse_canonical_wav

PARAMETERS_SCHEMA = "crux.oaf-smoke-generator-parameters/v1"
_TOP_LEVEL_KEYS = frozenset(
    {
        "bits_per_sample",
        "channel_count",
        "cymbal",
        "frame_count",
        "kick",
        "lcg",
        "sample_rate",
        "schema",
        "snare",
    }
)
_TRANSIENT_KEYS = frozenset({"amplitude", "decay_samples", "sample_index"})
_KICK_KEYS = frozenset(
    {
        "amplitude",
        "decay_samples",
        "period_end_samples",
        "period_start_samples",
        "sample_index",
    }
)
_LCG_KEYS = frozenset({"increment", "multiplier", "seed", "word_bits"})
_UINT32_MASK = (1 << 32) - 1

FIXED_PARAMETERS = {
    "bits_per_sample": 16,
    "channel_count": 1,
    "cymbal": {
        "amplitude": 12000,
        "decay_samples": 6615,
        "sample_index": 30870,
    },
    "frame_count": 44100,
    "kick": {
        "amplitude": 28000,
        "decay_samples": 2205,
        "period_end_samples": 180,
        "period_start_samples": 45,
        "sample_index": 4410,
    },
    "lcg": {
        "increment": 1013904223,
        "multiplier": 1664525,
        "seed": 0,
        "word_bits": 32,
    },
    "sample_rate": 44100,
    "schema": PARAMETERS_SCHEMA,
    "snare": {
        "amplitude": 20000,
        "decay_samples": 4410,
        "sample_index": 17640,
    },
}


class SmokeFixtureError(ValueError):
    """The requested smoke fixture does not match the frozen contract."""


def lcg_step(state: int, multiplier: int, increment: int) -> int:
    """Advance the explicit unsigned 32-bit LCG."""

    for value, field in (
        (state, "state"),
        (multiplier, "multiplier"),
        (increment, "increment"),
    ):
        if type(value) is not int or not 0 <= value <= _UINT32_MASK:
            raise SmokeFixtureError(f"LCG {field} must be an unsigned 32-bit integer")
    return (state * multiplier + increment) & _UINT32_MASK


def saturating_add(left: int, right: int) -> int:
    """Add two integer signals with signed-int16 saturation."""

    if type(left) is not int or type(right) is not int:
        raise SmokeFixtureError("PCM samples must be integers")
    return max(-32768, min(32767, left + right))


def generate_smoke_wav(parameters: Mapping[str, Any]) -> bytes:
    """Return exact RIFF/WAVE bytes for the frozen smoke parameters."""

    normalized = _validate_parameters(parameters)
    samples = [0] * normalized["frame_count"]
    _add_kick(samples, normalized["kick"])
    state = normalized["lcg"]["seed"]
    state = _add_noise_transient(samples, normalized["snare"], normalized["lcg"], state)
    _add_noise_transient(samples, normalized["cymbal"], normalized["lcg"], state)
    pcm = b"".join(struct.pack("<h", sample) for sample in samples)
    content = (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )
    parse_canonical_wav(content, max_input_audio_frames=44100)
    return content


def write_smoke_fixture(parameters_path: Path, output_path: Path) -> tuple[str, str]:
    """Generate and atomically publish the canonical parameters and WAV."""

    parameter_bytes = canonical_json_bytes(
        _plain_parameters(FIXED_PARAMETERS), trailing_newline=True
    )
    loaded = strict_json_loads(parameter_bytes)
    if not isinstance(loaded, dict):
        raise SmokeFixtureError("canonical smoke parameters must be an object")
    wav_bytes = generate_smoke_wav(loaded)
    atomic_replace_bytes(Path(parameters_path), parameter_bytes)
    atomic_replace_bytes(Path(output_path), wav_bytes)

    return (
        hashlib.sha256(parameter_bytes).hexdigest(),
        hashlib.sha256(wav_bytes).hexdigest(),
    )


def _validate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(parameters, Mapping) or set(parameters) != _TOP_LEVEL_KEYS:
        raise SmokeFixtureError("smoke parameters have unknown or missing fields")
    plain = _plain_parameters(parameters)
    if plain["schema"] != PARAMETERS_SCHEMA:
        raise SmokeFixtureError("smoke parameter schema is invalid")
    for field, expected in (
        ("sample_rate", 44100),
        ("channel_count", 1),
        ("bits_per_sample", 16),
        ("frame_count", 44100),
    ):
        if type(plain[field]) is not int or plain[field] != expected:
            raise SmokeFixtureError(f"{field} must equal {expected}")

    _validate_transient(plain["kick"], _KICK_KEYS, 4410, "kick")
    _validate_transient(plain["snare"], _TRANSIENT_KEYS, 17640, "snare")
    _validate_transient(plain["cymbal"], _TRANSIENT_KEYS, 30870, "cymbal")
    if set(plain["lcg"]) != _LCG_KEYS:
        raise SmokeFixtureError("LCG parameters have unknown or missing fields")
    for field in _LCG_KEYS:
        value = plain["lcg"][field]
        if type(value) is not int:
            raise SmokeFixtureError(f"LCG {field} must be an integer")
    if plain["lcg"]["word_bits"] != 32 or plain["lcg"]["seed"] != 0:
        raise SmokeFixtureError("LCG must use an unsigned 32-bit state seeded with zero")
    for field in ("multiplier", "increment"):
        if not 0 <= plain["lcg"][field] <= _UINT32_MASK:
            raise SmokeFixtureError(f"LCG {field} must be unsigned 32-bit")
    return plain


def _validate_transient(
    transient: object,
    expected_keys: frozenset[str],
    sample_index: int,
    name: str,
) -> None:
    if not isinstance(transient, dict) or set(transient) != expected_keys:
        raise SmokeFixtureError(f"{name} parameters have unknown or missing fields")
    for field, value in transient.items():
        if type(value) is not int or value <= 0:
            if field == "sample_index" and type(value) is int and value == 0:
                continue
            raise SmokeFixtureError(f"{name} {field} must be a positive integer")
    if transient["sample_index"] != sample_index:
        raise SmokeFixtureError(f"{name} sample_index must equal {sample_index}")
    if transient["sample_index"] + transient["decay_samples"] > 44100:
        raise SmokeFixtureError(f"{name} envelope exceeds the fixture")
    if transient["amplitude"] > 32767:
        raise SmokeFixtureError(f"{name} amplitude exceeds signed int16")
    if name == "kick":
        if transient["period_start_samples"] >= transient["period_end_samples"]:
            raise SmokeFixtureError("kick period must increase through the envelope")


def _add_kick(samples: list[int], kick: dict[str, int]) -> None:
    start = kick["sample_index"]
    duration = kick["decay_samples"]
    phase = 0
    polarity = 1
    for offset in range(duration):
        remaining = duration - offset
        amplitude = kick["amplitude"] * remaining // duration
        period = kick["period_start_samples"] + (
            (kick["period_end_samples"] - kick["period_start_samples"]) * offset // duration
        )
        if phase >= period:
            phase = 0
            polarity = -polarity
        samples[start + offset] = saturating_add(
            samples[start + offset],
            polarity * amplitude,
        )
        phase += 1


def _add_noise_transient(
    samples: list[int],
    transient: dict[str, int],
    lcg: dict[str, int],
    state: int,
) -> int:
    start = transient["sample_index"]
    duration = transient["decay_samples"]
    for offset in range(duration):
        state = lcg_step(state, lcg["multiplier"], lcg["increment"])
        signed_noise = ((state >> 16) & 0xFFFF) - 32768
        remaining = duration - offset
        value = signed_noise * transient["amplitude"] * remaining // (32768 * duration)
        samples[start + offset] = saturating_add(samples[start + offset], value)
    return state


def _plain_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in parameters.items():
        result[key] = _plain_parameters(value) if isinstance(value, Mapping) else value
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        parameter_hash, wav_hash = write_smoke_fixture(args.parameters, args.output)
    except (OSError, SmokeFixtureError, TypeError, ValueError) as error:
        print(f"smoke fixture generation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"parameters_sha256": parameter_hash, "wav_sha256": wav_hash},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
