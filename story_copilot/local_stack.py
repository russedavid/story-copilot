"""Run public application/model/speech entry points with an external private configuration."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen
from urllib.parse import urlsplit
from uuid import uuid4

MODULES = {
    "model": "qwen_ttrpg.serve_models",
    "planner": "qwen_ttrpg.serve_models",
    "speech": "story_copilot.audio_worker",
    "app": "story_copilot.cli",
}


def configuration(path):
    config = json.loads(Path(path).expanduser().read_text())
    root = Path(config["run_root"]).expanduser().resolve()
    if any((p / ".git").exists() for p in [root, *root.parents]):
        raise ValueError("Runtime state and logs belong outside source repositories.")
    services = config["services"]
    names = [s["name"] for s in services]
    if not services or len(set(names)) != len(names) or set(names) - set(MODULES):
        raise ValueError("Use unique model, planner, speech and app service names.")
    if names[-1] != "app" or "model" not in names:
        raise ValueError("Start the model before the application; put app last.")
    for service in services:
        argv = service["argv"]
        if (
            not isinstance(argv, list)
            or len(argv) < 3
            or not all(isinstance(x, str) for x in argv)
            or argv[1:3] != ["-m", MODULES[service["name"]]]
            or not Path(argv[0]).is_file()
            or not Path(service["cwd"]).is_dir()
            or not (
                Path(service["cwd"])
                / (MODULES[service["name"]].replace(".", "/") + ".py")
            ).is_file()
        ):
            raise ValueError(
                "Use an existing Python executable, public module, and release directory."
            )
        url = service.get("health_url")
        if url and (
            urlsplit(url).scheme != "http"
            or urlsplit(url).hostname not in {"localhost", "127.0.0.1", "::1"}
            or urlsplit(url).username
            or urlsplit(url).password
        ):
            raise ValueError("Service health checks must use local HTTP endpoints.")
        if not url and not service.get("ready_text"):
            raise ValueError(
                "Each service needs health_url or a ready_text log marker."
            )
    return root, services


def healthy(url):
    try:
        with urlopen(url, timeout=2) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def shutdown(children):
    for child, service in reversed(children):
        try:
            os.killpg(
                child.pid,
                signal.SIGINT if service["name"] == "speech" else signal.SIGTERM,
            )
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    for child, _ in reversed(children):
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
    for child, _ in children:
        # A failed supervisor may exit before its model subprocess does.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(path):
    root, services = configuration(path)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = (root / "stack.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Refuse occupied configured endpoints before loading any additional weights.
    for service in services:
        if not service.get("health_url"):
            continue
        endpoint = urlsplit(service["health_url"])
        try:
            connection = socket.create_connection(
                (endpoint.hostname, endpoint.port or 80), timeout=1
            )
        except OSError:
            continue
        connection.close()
        raise RuntimeError(
            "A configured endpoint is already running; stop its existing owner first."
        )
    directory = root / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    directory.mkdir(mode=0o700)
    state = {"status": "starting", "run": str(directory), "services": {}}
    children = []
    stop = False

    def interrupted(signum, frame):
        nonlocal stop
        stop = True

    previous = {
        sig: signal.signal(sig, interrupted) for sig in [signal.SIGINT, signal.SIGTERM]
    }

    def persist():
        (directory / "status.json").write_text(json.dumps(state, indent=2))
        temporary = root / "current.tmp"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(root / "current.json")

    try:
        for service in services:
            if stop:
                break
            argv = [value.replace("{run}", str(directory)) for value in service["argv"]]
            env = os.environ.copy()
            env.update(
                {
                    k: str(v).replace("{run}", str(directory))
                    for k, v in service.get("env", {}).items()
                }
            )
            env["PYTHONUNBUFFERED"] = "1"
            log = directory / (service["name"] + ".log")
            with log.open("w") as stream:
                child = subprocess.Popen(
                    argv,
                    cwd=service["cwd"],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            children.append((child, service))
            state["services"][service["name"]] = {
                "pid": child.pid,
                "cwd": service["cwd"],
                "module": MODULES[service["name"]],
                "revision": service.get("revision", "not specified"),
                "status": "starting",
            }
            persist()
            deadline = time.monotonic() + 180
            while not stop:
                if any(p.poll() is not None for p, _ in children):
                    raise RuntimeError("A service exited; inspect its private log.")
                ready = (
                    healthy(service["health_url"])
                    if service.get("health_url")
                    else service["ready_text"] in log.read_text(errors="replace")
                )
                if ready:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Service startup timed out; inspect its private log."
                    )
                time.sleep(0.25)
            if stop:
                break
            state["services"][service["name"]]["status"] = "ready"
            persist()
        if not stop:
            state["status"] = "ready"
            persist()
            print(json.dumps({"status": "ready", "run": str(directory)}), flush=True)
        while not stop:
            if any(p.poll() is not None for p, _ in children):
                raise RuntimeError("A service exited; stopping the owned stack.")
            time.sleep(0.5)
        state["status"] = "stopped"
    except BaseException as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        shutdown(children)
        for row in state["services"].values():
            row["status"] = "stopped"
        persist()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Private JSON configuration outside the repository.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
