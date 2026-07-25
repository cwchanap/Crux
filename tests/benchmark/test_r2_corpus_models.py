from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from src.benchmark.r2_corpus_models import (
    CACHE_PROFILE,
    MAX_SIMFILE_ID,
    R2Config,
    SyncRequest,
    format_manifest_timestamp,
    format_report_filename_timestamp,
    parse_etag,
)


def test_config_normalizes_endpoint_and_hashes_only_normalized_origin():
    config = R2Config.from_environ(
        {
            "CRUX_R2_ENDPOINT_URL": "HTTPS://ABC123.r2.cloudflarestorage.com/",
            "CRUX_R2_BUCKET": "simfile-dtx",
        }
    )
    normalized = "https://abc123.r2.cloudflarestorage.com"
    assert config.endpoint_url == normalized
    assert config.source_endpoint_sha256 == sha256(normalized.encode("ascii")).hexdigest()
    assert config.bucket == "simfile-dtx"
    assert config.head_concurrency == 8
    assert config.download_concurrency == 4
    assert config.connect_timeout_seconds == 10
    assert config.read_timeout_seconds == 60
    assert config.max_attempts == 5


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://abc.r2.cloudflarestorage.com",
        "https://user@example.com",
        "https://example.com/path",
        "https://example.com?token=value",
        "https://example.com#fragment",
    ],
)
def test_config_rejects_non_https_origin(endpoint):
    with pytest.raises(ValueError, match="HTTPS origin"):
        R2Config.from_environ({"CRUX_R2_ENDPOINT_URL": endpoint})


@pytest.mark.parametrize(
    ("environ", "variable"),
    [
        ({}, "CRUX_R2_ENDPOINT_URL"),
        ({"CRUX_R2_ENDPOINT_URL": "https://example.com", "CRUX_R2_BUCKET": ""}, "CRUX_R2_BUCKET"),
        (
            {"CRUX_R2_ENDPOINT_URL": "https://example.com", "CRUX_R2_BUCKET": "bad/name"},
            "CRUX_R2_BUCKET",
        ),
        (
            {"CRUX_R2_ENDPOINT_URL": "https://example.com", "CRUX_R2_HEAD_CONCURRENCY": "0"},
            "CRUX_R2_HEAD_CONCURRENCY",
        ),
        (
            {"CRUX_R2_ENDPOINT_URL": "https://example.com", "CRUX_R2_HEAD_CONCURRENCY": "33"},
            "CRUX_R2_HEAD_CONCURRENCY",
        ),
        (
            {"CRUX_R2_ENDPOINT_URL": "https://example.com", "CRUX_R2_DOWNLOAD_CONCURRENCY": "0"},
            "CRUX_R2_DOWNLOAD_CONCURRENCY",
        ),
        (
            {"CRUX_R2_ENDPOINT_URL": "https://example.com", "CRUX_R2_DOWNLOAD_CONCURRENCY": "17"},
            "CRUX_R2_DOWNLOAD_CONCURRENCY",
        ),
        (
            {"CRUX_R2_ENDPOINT_URL": "https://example.com", "CRUX_R2_CONNECT_TIMEOUT_SECONDS": "0"},
            "CRUX_R2_CONNECT_TIMEOUT_SECONDS",
        ),
        (
            {"CRUX_R2_ENDPOINT_URL": "https://example.com", "CRUX_R2_READ_TIMEOUT_SECONDS": "zero"},
            "CRUX_R2_READ_TIMEOUT_SECONDS",
        ),
        (
            {"CRUX_R2_ENDPOINT_URL": "https://example.com", "CRUX_R2_MAX_ATTEMPTS": "-1"},
            "CRUX_R2_MAX_ATTEMPTS",
        ),
    ],
)
def test_config_errors_are_sanitized(environ, variable):
    with pytest.raises(ValueError) as error:
        R2Config.from_environ(environ)
    assert variable in str(error.value)
    for value in environ.values():
        if value:
            assert value not in str(error.value)


def test_config_conversion_errors_do_not_retain_the_supplied_value():
    secret = "signed-url-secret"
    with pytest.raises(ValueError) as error:
        R2Config.from_environ(
            {
                "CRUX_R2_ENDPOINT_URL": "https://example.com",
                "CRUX_R2_READ_TIMEOUT_SECONDS": secret,
            }
        )
    assert error.value.__cause__ is None
    assert secret not in str(error.value)


@pytest.mark.parametrize(
    ("field_name", "simfile_id"),
    [("include_simfile_ids", -1), ("exclude_simfile_ids", MAX_SIMFILE_ID + 1)],
)
def test_sync_request_rejects_out_of_range_simfile_ids(field_name, simfile_id):
    with pytest.raises(ValueError, match="simfile ID"):
        SyncRequest(
            output_dir=Path("output"),
            cache_dir=Path("cache"),
            provenance_file=None,
            **{field_name: frozenset({simfile_id})},
        )


def test_etag_and_timestamp_normalization_are_canonical():
    assert parse_etag('"abc-2"') == ("abc-2", False)
    assert parse_etag('W/"weak-value"') == ("weak-value", True)
    with pytest.raises(ValueError, match="entity tag"):
        parse_etag("not-quoted")
    with pytest.raises(ValueError, match="entity tag"):
        parse_etag('"contains\ncontrol"')
    value = datetime(2026, 7, 25, 1, 2, 3, 120000, tzinfo=timezone.utc)
    assert format_manifest_timestamp(value) == "2026-07-25T01:02:03.12Z"
    assert format_report_filename_timestamp(value) == "20260725T010203.120000Z"


def test_fixed_contract_constants_are_stable():
    assert CACHE_PROFILE == "setdef_dtx_txt_v1"
    assert MAX_SIMFILE_ID == 9_007_199_254_740_991
