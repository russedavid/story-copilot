import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

from story_copilot.local_stack import configuration, run


def available_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def fixture_config(tmp_path, *, fail=False):
    source = tmp_path / "public-release"
    for package in ["qwen_ttrpg", "story_copilot"]:
        (source / package).mkdir(parents=True)
        (source / package / "__init__.py").write_text("")
    (source / "qwen_ttrpg/serve_models.py").write_text(
        """from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers()
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    )
    (source / "story_copilot/cli.py").write_text(
        """import time,sys
from pathlib import Path
print('APP READY',flush=True)
if len(sys.argv)>1:
 while not Path(sys.argv[1]).exists():time.sleep(0.05)
 raise RuntimeError('fixture failure')
while True:time.sleep(0.1)
"""
    )
    port = available_port()
    config = {
        "run_root": str(tmp_path / "private-runs"),
        "services": [
            {
                "name": "model",
                "cwd": str(source),
                "argv": [sys.executable, "-m", "qwen_ttrpg.serve_models", str(port)],
                "health_url": f"http://127.0.0.1:{port}/health",
            },
            {
                "name": "app",
                "cwd": str(source),
                "argv": [
                    sys.executable,
                    "-m",
                    "story_copilot.cli",
                    *([str(tmp_path / "fail")] if fail else []),
                ],
                "ready_text": "APP READY",
            },
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config, port


def status(root, child, target="ready"):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        path = Path(root) / "current.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["status"] == target:
                return result
        if child.poll() is not None:
            break
        time.sleep(0.05)
    raise AssertionError("Stack did not reach " + target)


@pytest.mark.parametrize("fail", [False, True])
def test_owned_stack_readiness_shutdown_and_failure_cleanup(tmp_path, fail):
    path, config, port = fixture_config(tmp_path, fail=fail)
    child = subprocess.Popen(
        [sys.executable, "-m", "story_copilot.local_stack", "--config", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        ready = status(config["run_root"], child)
        assert ready["supervisor_pid"] == child.pid
        if fail:
            (tmp_path / "fail").touch()
        else:
            child.send_signal(signal.SIGTERM)
        code = child.wait(timeout=10)
        assert (code != 0) == fail
        final = json.loads((Path(config["run_root"]) / "current.json").read_text())
        assert final["status"] == ("failed" if fail else "stopped")
        for service in ready["services"].values():
            with pytest.raises(ProcessLookupError):
                os.kill(service["pid"], 0)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.1)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=25)


def test_refuses_existing_listener_without_stopping_it(tmp_path):
    path, config, port = fixture_config(tmp_path)
    with socket.socket() as existing:
        existing.bind(("127.0.0.1", port))
        existing.listen()
        with pytest.raises(RuntimeError, match="already running"):
            run(path)
        assert existing.fileno() >= 0
    assert not (Path(config["run_root"]) / "current.json").exists()


def test_requires_public_module_and_private_output_location(tmp_path):
    path, config, port = fixture_config(tmp_path)
    config["services"][0]["argv"][1:3] = ["-c", "print(1)"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="public module"):
        configuration(path)
    path, config, _ = fixture_config(tmp_path / "other")
    root = Path(config["run_root"])
    root.mkdir()
    (root / ".git").mkdir()
    with pytest.raises(ValueError, match="outside source"):
        configuration(path)
