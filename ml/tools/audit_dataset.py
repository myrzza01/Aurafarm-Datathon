"""Repeat local numeric duplicate/overlap audit; exports metadata only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cashwatch.data import audit_sequences, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="artifacts/data/manifest.json")
    parser.add_argument("--output", default="artifacts/data/group_audit.json")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    sequences = {}
    records = manifest["records"] + manifest.get("quarantined_records", [])
    for record in records:
        with np.load(record["cache"], allow_pickle=False) as saved:
            images = saved["images"]
            sequences[record["id"]] = (images.reshape(len(images), 16, 14, 16, 14, 3).mean(axis=(2, 4)).astype(np.float32), saved["times"])
    report = audit_sequences(records, sequences)
    write_json(args.output, report)
    print(json.dumps({"pairs_compared": report["pairs_compared"],
                      "exact_pairs": len(report["exact_byte_pairs"]),
                      "confirmed_overlap_pairs": len(report["confirmed_overlap_pairs"]),
                      "possible_related_ids": len(report["possible_related_ids"]),
                      "manual_source_review_complete": False}))


if __name__ == "__main__":
    main()
