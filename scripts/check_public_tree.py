"""Check tracked files (or reachable history) without printing private matches.

Supply an external --terms-file containing one private identifier per line for a
release audit. Never commit that list or the audit's private working directory.
Git author/committer identity is intentional attribution; messages remain scanned.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

FORBIDDEN_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".mp4",
    ".mov",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".epub",
    ".safetensors",
    ".gguf",
    ".pt",
    ".pth",
    ".bin",
    ".sqlite",
    ".db",
    ".jsonl",
    ".ipynb",
}
FORBIDDEN_PARTS = {
    "private",
    "datasets",
    "outputs",
    "runs",
    "models",
    "checkpoints",
    ".venv",
}
PATTERNS = {
    "private absolute path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private IPv4 address": re.compile(
        r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"
    ),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "provider credential": re.compile(
        r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gsk_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{25,}|AIza[A-Za-z0-9_-]{30,})\b"
    ),
    "personal email": re.compile(
        r"\b[A-Za-z0-9._%+-]+@(?:gmail|hotmail|outlook|yahoo)\.com\b", re.I
    ),
}


def git(*args):
    return subprocess.check_output(["git", *args])


def check(path, payload, terms):
    problems = []
    location = Path(path)
    if (
        location.suffix.lower() in FORBIDDEN_SUFFIXES
        or set(location.parts) & FORBIDDEN_PARTS
    ):
        problems.append("private/generated file type or directory")
    if location.name == ".env" or (
        location.name.startswith(".env.") and location.name != ".env.example"
    ):
        problems.append("environment file")
    if len(payload) > 2_000_000:
        problems.append("unexpected large file")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return problems + ["binary content"]
    for name, pattern in PATTERNS.items():
        if pattern.search(text):
            problems.append(name)
    if any(term in (path + "\n" + text).casefold() for term in terms):
        problems.append("private identifier")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--terms-file")
    args = parser.parse_args()
    terms = (
        [
            s.strip().casefold()
            for s in Path(args.terms_file).read_text().splitlines()
            if s.strip()
        ]
        if args.terms_file
        else []
    )
    failures, count = [], 0
    if args.history:
        objects = (
            git("rev-list", "--objects", "--branches", "--tags").decode().splitlines()
        )
        records = []
        for item in objects:
            oid, _, path = item.partition(" ")
            if path and git("cat-file", "-t", oid).strip() == b"blob":
                records.append((path, git("cat-file", "blob", oid)))
        # Attribution belongs in Git's author/committer fields. Continue checking
        # messages for private source names, paths, and secrets.
        messages = git("log", "--branches", "--tags", "--format=%B")
        records.append(("commit-messages", messages))
    else:
        records = [
            (p, Path(p).read_bytes())
            for p in git("ls-files", "-z").decode().split("\0")
            if p and Path(p).is_file()
        ]
    for path, payload in records:
        count += 1
        issues = check(path, payload, terms)
        if issues:
            # Paths themselves may identify source material; print only category counts.
            failures.extend(issues)
    print(
        json.dumps(
            {
                "checked": count,
                "findings": {s: failures.count(s) for s in sorted(set(failures))},
            }
        )
    )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
