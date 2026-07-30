from __future__ import annotations

import json
import os
import signal
import sys
import time

PROTOCOL = "crux.transcription-runner/v1"


def _line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _write(value: object) -> None:
    sys.stdout.buffer.write(_line(value))
    sys.stdout.buffer.flush()


def _ready() -> None:
    _write({"protocol_schema": PROTOCOL, "type": "ready"})


def main() -> int:
    mode = sys.argv[1]
    if mode == "startup_timeout":
        time.sleep(30)
        return 0
    if mode == "startup_death":
        return 17
    if mode == "startup_malformed":
        sys.stdout.buffer.write(b"{bad\n")
        sys.stdout.buffer.flush()
        return 0
    if mode == "startup_stray":
        sys.stdout.buffer.write(b"not protocol\n")
        sys.stdout.buffer.flush()
        return 0
    if mode == "startup_eof":
        sys.stdout.buffer.write(b'{"type":"ready"}')
        sys.stdout.buffer.flush()
        return 0
    if mode == "startup_oversized":
        sys.stdout.buffer.write(b"x" * 4096)
        sys.stdout.buffer.flush()
        time.sleep(30)
        return 0
    if mode == "startup_flood":
        ready = json.dumps(
            {"protocol_schema": PROTOCOL, "type": "ready"},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        line = b'{"request_id":"stray","type":"response"}\n'
        sys.stdout.buffer.write(ready + b"\n" + line * 65536)
        sys.stdout.buffer.flush()
        time.sleep(30)
        return 0
    if mode == "delayed_startup_stray":
        _ready()
        time.sleep(0.15)
        sys.stdout.buffer.write(b"delayed stray stdout\n")
        sys.stdout.buffer.flush()
        time.sleep(30)
        return 0
    if mode == "delayed_oversized":
        _ready()
        time.sleep(0.15)
        sys.stdout.buffer.write(b"x" * 4096)
        sys.stdout.buffer.flush()
        time.sleep(30)
        return 0
    if mode == "delayed_partial_eof":
        _ready()
        time.sleep(0.15)
        sys.stdout.buffer.write(b'{"type":"response"')
        sys.stdout.buffer.flush()
        return 0
    if mode == "delayed_clean_eof":
        _ready()
        time.sleep(0.15)
        return 0
    if mode == "busy_stderr":
        _ready()
        os.write(2, b"code=runner_waiting count=1\n")
        time.sleep(30)
        return 0

    _ready()
    request_line = sys.stdin.buffer.readline()
    if not request_line:
        return 0
    request = json.loads(request_line)
    request_id = request["request_id"]
    if mode == "request_timeout":
        time.sleep(30)
        return 0
    if mode == "request_death":
        return 19
    if mode == "request_malformed":
        sys.stdout.buffer.write(b"{bad\n")
        sys.stdout.buffer.flush()
        return 0
    if mode == "request_wrong_id":
        _write({"payload": {"type": "pong"}, "request_id": "wrong", "type": "response"})
        return 0
    if mode == "request_eof":
        sys.stdout.buffer.write(
            json.dumps(
                {"payload": {"type": "pong"}, "request_id": request_id, "type": "response"},
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        sys.stdout.buffer.flush()
        return 0
    if mode == "request_oversized":
        sys.stdout.buffer.write(b"x" * 4096)
        sys.stdout.buffer.flush()
        time.sleep(30)
        return 0
    if mode == "stderr_stress":
        os.write(2, b"code=runner_stress count=1\n" * 65536)
    if mode == "stderr_sensitive":
        fragments = (
            b"token=super",
            b"-secret\n/path=/private/secret/model.ckpt\n",
            b"url=https://user:pass@example.test/model?token=x#fragment\n",
            b"audio=[0.123,0.456,0.789,0.012,0.345,0.678]\n",
            b'Traceback (most recent call last):\n  File "/secret/a.py", line 1\n',
            b"code=restore_ok tensor=onsets/count count=78 duration_ms=42\n",
        )
        for fragment in fragments:
            os.write(2, fragment)
    if mode == "stderr_unterminated":
        os.write(2, b"secret=" + b"x" * 16384)
    if mode == "stderr_paths":
        fragments = (
            b"code=restore_ok tensor=/unknown/private/model.ckpt count=78 ",
            b"duration_ms=42\n",
            b"code=restore_ok tensor=C:\\Users\\alice\\model.ckpt count=78 duration_ms=42\n",
            b"code=restore_ok tensor=\\\\server\\share\\model.ckpt count=78 duration_ms=42\n",
            b"code=restore_ok tensor=file:///unknown/private/model.ckpt count=78 duration_ms=42\n",
            b"code=restore_ok tensor=https://example.test/private/model.ckpt count=78 ",
            b"duration_ms=42\n",
            b"code=restore_ok tensor=onsets/conv/kernel:0 count=78 duration_ms=42\n",
        )
        for fragment in fragments:
            os.write(2, fragment)
    if mode == "ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "hold":
        time.sleep(0.5)
    response = {"payload": {"type": "pong"}, "request_id": request_id, "type": "response"}
    if mode == "request_duplicate":
        compact_response = {"payload": {}, "request_id": request_id, "type": "response"}
        os.write(1, _line(compact_response) + _line(compact_response))
        time.sleep(30)
        return 0
    if mode == "request_extra":
        os.write(1, _line(response) + _line({"type": "unexpected"}))
        time.sleep(30)
        return 0
    _write(response)
    if mode == "delayed_request_duplicate":
        time.sleep(0.15)
        _write(response)
        time.sleep(30)
        return 0
    if mode == "ignore_term":
        time.sleep(30)
    if mode in {
        "success",
        "hold",
        "stderr_stress",
        "stderr_sensitive",
        "stderr_unterminated",
        "stderr_paths",
    }:
        # Successful protocol fixtures model the persistent production runner.
        # The host closes stdin during graceful teardown.
        sys.stdin.buffer.read()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
