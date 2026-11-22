from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cloudsce.trace import (SUBSET_DIR, TraceRecord, synthesize_subset,
                            train_record_from_csv)


def to_json(name: str, kind: str, records: list, meta: dict) -> dict:
    return {
        "name": name,
        "kind": kind,
        "meta": meta,
        "records": [
            {"cid": int(r.cid), "frac": [float(x) for x in r.frac],
             "n_inst": int(r.n_inst), "dur": float(r.dur),
             "slack": float(r.slack), "latency": float(r.latency)}
            for r in records
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default=None,
                    help="path to the real batch_task.csv of cluster-trace-v2018")
    ap.add_argument("--sample-frac", type=float, default=0.02)
    ap.add_argument("--records", type=int, default=4096)
    ap.add_argument("--synthesize", action="store_true",
                    help="build the statistical twin when the raw trace is absent")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(SUBSET_DIR, exist_ok=True)
    if args.raw:
        recs = train_record_from_csv(args.raw, args.sample_frac)
        name, kind = "alibaba", "alibaba"
        meta = {"source": os.path.basename(args.raw), "sample_frac": args.sample_frac}
    else:
        sub = synthesize_subset("alibaba", "alibaba", args.records)
        recs, name, kind = sub.records, "alibaba", "alibaba"
        meta = {"source": "statistical-twin", "records": len(recs),
                "note": "generated because the raw cluster-trace-v2018 files "
                        "were not available offline"}

    for nm, kd in (("alibaba", "alibaba"), ("google", "google")):
        if nm == name:
            payload = to_json(name, kind, recs, meta)
        else:
            payload = to_json(nm, kd, synthesize_subset(nm, kd, args.records).records,
                              {"source": "statistical-twin", "records": args.records})
        path = args.out if (args.out and nm == "alibaba") else os.path.join(SUBSET_DIR, f"{nm}_records.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        print(f"wrote {path}  ({len(payload['records'])} records)")

    fracs = np.asarray([r.frac for r in recs])
    print("\nper-resource mean demand (fraction of a reference node):")
    for i, k in enumerate(("cpu", "npu", "vram", "mem", "disk", "bw")):
        print(f"  {k:5s} {fracs[:, i].mean():.3f} +- {fracs[:, i].std():.3f}")
    print(f"replicas  mean={np.mean([r.n_inst for r in recs]):.2f} "
          f"max={max(r.n_inst for r in recs)}")
    print(f"duration  mean={np.mean([r.dur for r in recs]):.2f} rounds")


if __name__ == "__main__":
    main()
