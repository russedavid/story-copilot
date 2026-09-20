"""Locate speaker disagreements for review; agreement is not an accuracy metric."""

import argparse
from collections import Counter
from itertools import permutations
import json
from pathlib import Path


def compare(left, right):
    a = [w for s in left["segments"] for w in s.get("words", [])]
    b = [w for s in right["segments"] for w in s.get("words", [])]
    if len(a) != len(b) or any(
        (x.get("word"), x.get("start"), x.get("end"))
        != (y.get("word"), y.get("start"), y.get("end"))
        for x, y in zip(a, b)
    ):
        raise ValueError(
            "Compare the same aligned words; retranscribed words require a separate alignment."
        )
    label = lambda w: w.get("speaker") or "Unknown"
    labels_a = sorted({label(w) for w in a} - {"Unknown"})
    labels_b = sorted({label(w) for w in b} - {"Unknown"})
    overlaps = Counter((label(x), label(y)) for x, y in zip(a, b))
    padded = labels_a + [
        f"UNMATCHED_{i}" for i in range(max(0, len(labels_b) - len(labels_a)))
    ]
    if max(len(labels_a), len(labels_b)) > 8:
        raise ValueError("This pilot comparer supports up to eight speakers.")
    best = max(
        permutations(padded, len(labels_b)),
        key=lambda p: sum(overlaps[(x, y)] for x, y in zip(p, labels_b)),
        default=(),
    )
    mapping = dict(zip(labels_b, best))
    disagreements = []
    both = 0
    unknown_a = 0
    unknown_b = 0
    for index, (x, y) in enumerate(zip(a, b)):
        la, lb = label(x), label(y)
        unknown_a += la == "Unknown"
        unknown_b += lb == "Unknown"
        if la == "Unknown" or lb == "Unknown":
            continue
        both += 1
        if la != mapping.get(lb):
            disagreements.append(
                {
                    "word_index": index,
                    "word": x["word"],
                    "start": x.get("start"),
                    "end": x.get("end"),
                    "left_speaker": la,
                    "right_speaker": lb,
                    "right_mapped_to": mapping.get(lb),
                }
            )
    spans = []
    for word in disagreements:
        if word["start"] is None or word["end"] is None:
            continue
        if (
            spans
            and word["start"] - spans[-1]["end"] < 0.8
            and word["end"] - spans[-1]["start"] <= 15
            and (word["left_speaker"], word["right_speaker"])
            == (spans[-1]["left_speaker"], spans[-1]["right_speaker"])
        ):
            spans[-1]["end"] = word["end"]
            spans[-1]["last_word"] = word["word_index"]
        else:
            spans.append(
                {
                    **word,
                    "first_word": word["word_index"],
                    "last_word": word["word_index"],
                }
            )
    for span in spans:
        span["context"] = " ".join(
            w["word"] for w in a[max(0, span["first_word"] - 8) : span["last_word"] + 9]
        )
    return {
        "purpose": "Locate disagreements for listening review. These are not DER, correctness, or a model ranking.",
        "aligned_words": len(a),
        "assigned_by_both": both,
        "unassigned_left": unknown_a,
        "unassigned_right": unknown_b,
        "speaker_mapping_right_to_left": mapping,
        "disagreed_words": len(disagreements),
        "disagreement_fraction": len(disagreements) / both if both else None,
        "left_provenance": left.get("provenance", {}),
        "right_provenance": right.get("provenance", {}),
        "disagreements": disagreements,
        "review_spans": spans,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("output")
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Choose a new comparison output.")
    result = compare(
        json.loads(Path(args.left).read_text()),
        json.loads(Path(args.right).read_text()),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in {
                    "disagreements",
                    "review_spans",
                    "left_provenance",
                    "right_provenance",
                }
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
