"""Resume an episode manifest without repeating completed audio processing."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("output_dir")
    p.add_argument("--device-index", type=int, default=1)
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for job in manifest:
        jid = job["id"]
        if Path(jid).name != jid or jid in {".", ".."} or "/" in jid or "\\" in jid:
            raise ValueError("Episode IDs must be simple filenames.")
        path = output_dir / (jid + ".json")
        aligned = None
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("provenance", {}).get("diarization") == "complete":
                print("COMPLETE", jid, flush=True)
                results.append({"id": jid, "status": "complete", "path": str(path)})
                continue
            aligned = path.with_name(
                path.stem + ".aligned-" + uuid4().hex[:8] + ".json"
            )
            path.rename(aligned)
        cmd = [
            sys.executable,
            "-m",
            "story_copilot.asr",
            job["audio"],
            str(path),
            "--seconds",
            str(job["seconds"]),
            "--device-index",
            str(args.device_index),
        ]
        if job.get("speakers"):
            cmd += ["--speakers", str(job["speakers"])]
        if aligned:
            cmd += ["--aligned", str(aligned)]
        env = os.environ.copy()
        libs = sorted(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/*/lib"))
        if libs:
            env["LD_LIBRARY_PATH"] = ":".join(map(str, libs))
        print("PROCESSING", jid, flush=True)
        with path.with_suffix(".log").open("w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        doc = json.loads(path.read_text()) if path.exists() else {}
        ok = (
            proc.returncode == 0
            and doc.get("provenance", {}).get("diarization") == "complete"
        )
        result = {
            "id": jid,
            "status": "complete" if ok else "failed",
            "path": str(path),
            "returncode": proc.returncode,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        (output_dir / "batch-status.json").write_text(json.dumps(results, indent=2))
    (output_dir / "batch-status.json").write_text(json.dumps(results, indent=2))
    if any(r["status"] == "failed" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
