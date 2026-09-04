from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.diagnostics.transfer_audit import audit_edge_csv, audit_hdf, audit_npz_array
from stanchor.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a cross-dataset traffic series and graph.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", required=True, help="HDF or NPZ traffic file.")
    parser.add_argument("--graph", default=None, help="Optional edge CSV; HDF graph is audited separately by the loader.")
    parser.add_argument("--npz-key", default="data")
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_path = Path(args.data)
    if data_path.suffix.lower() == ".npz":
        result = audit_npz_array(data_path, key=args.npz_key)
    elif data_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
        result = audit_hdf(data_path)
    else:
        raise ValueError("data must be .h5/.hdf/.hdf5 or .npz")
    if args.graph is not None:
        if args.num_nodes is None:
            args.num_nodes = int(result["nodes"])
        result["graph"] = audit_edge_csv(args.graph, args.num_nodes)
    result["dataset"] = args.dataset
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    save_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"audit output: {output.resolve()}")


if __name__ == "__main__":
    main()

