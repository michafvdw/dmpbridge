"""Fetch a DMP from dmponline and save stage-1 extracted blocks.

Example
-------
python scripts/fetch_dmponline.py --url "https://dmponline.org/planner/12345" --sample 1

The script writes: `data/output/1_extracted/dmponline/sample{N}.json`
so it can be used directly by the existing pipeline (set --extractor dmponline).
"""
import argparse
import json
from pathlib import Path

from dmpbridge.extractors import get_extractor
from dmpbridge.core import paths


def main():
    ap = argparse.ArgumentParser(description="Fetch DMP from dmponline and save extracted blocks")
    ap.add_argument("--url", required=True, help="Public or private URL of the DMP page")
    ap.add_argument("--sample", type=int, default=1, help="Sample index to save as (default: 1)")
    ap.add_argument("--cookie", default=None, help="Session cookie string for authenticated DMPOnline access")
    args = ap.parse_args()

    # Allow passing a cookie on the command line; if provided set env var used by the extractor
    if args.cookie:
        import os
        os.environ["DMPONLINE_COOKIE"] = args.cookie

    extractor = get_extractor("dmponline")
    blocks = extractor.extract(Path(args.url))

    out_dir = paths.EXTRACTED_DIR / "dmponline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sample{args.sample}.json"
    out_path.write_text(json.dumps(blocks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(blocks)} blocks -> {out_path}")


if __name__ == "__main__":
    main()
