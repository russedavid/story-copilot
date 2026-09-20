"""Record source-preserving target reviews independently of model scores."""

import argparse
import json
from pathlib import Path

from .store import Store, digest


def curate(store, cid, reviews, reviewer):
    if not reviewer.strip():
        raise ValueError("Name the reviewer; assistant review is not human gold.")
    turns = store.turns(cid)
    lookup = {t["ordinal"]: t for t in turns}
    seen = set()
    for review in reviews:
        key = tuple(review["target_turns"])
        if (
            not key
            or key in seen
            or list(key) != list(range(key[0], key[-1] + 1))
            or any(n not in lookup for n in key)
        ):
            raise ValueError("Unknown or duplicate target group.")
        seen.add(key)
        if (
            review["verdict"] not in {"keep", "reject", "uncertain"}
            or not review["reason"].strip()
        ):
            raise ValueError("A reasoned verdict is required.")
        if any(
            lookup[n]["role"] != "facilitator"
            or lookup[n]["speaker"] != lookup[key[0]]["speaker"]
            for n in key
        ):
            raise ValueError("Review a contiguous Facilitator speaker span.")
        if review["text_sha256"] != digest(" ".join(lookup[n]["text"] for n in key)):
            raise ValueError("Reviewed text differs from the current target.")
    request = {
        "source_revisions": {t["id"]: t["revision"] for t in turns},
        "reviewer": reviewer,
    }
    rid = store.start_run(cid, "target_curation", request)
    store.finish_run(
        rid,
        {
            "reviews": reviews,
            "reviewer": reviewer,
            "scope": "Text/attribution review; audio not verified. Assistant review is not human-calibrated gold.",
        },
    )
    return rid


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--home", required=True)
    p.add_argument("collection")
    p.add_argument("reviews")
    p.add_argument("--reviewer", required=True)
    args = p.parse_args()
    print(
        curate(
            Store(args.home),
            args.collection,
            json.loads(Path(args.reviews).read_text()),
            args.reviewer,
        )
    )
