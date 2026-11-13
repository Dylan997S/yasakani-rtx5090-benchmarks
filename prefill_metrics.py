"""Join HTTP measurements to cold-server engine timings in request order."""

import argparse
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("ninfer", "yasakani"))
    parser.add_argument("client", type=Path, help="JSONL from bench.py --suite prefill")
    parser.add_argument("server", type=Path, help="Dedicated server log or NInfer request JSONL")
    args = parser.parse_args()
    client = [json.loads(line) for line in args.client.read_text(encoding="utf-8-sig").splitlines()]
    raw = args.server.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    lines = raw.decode(encoding, errors="replace").splitlines()
    events = [json.loads(line) for line in lines if line.startswith("{")]
    if args.engine == "ninfer":
        native = [
            (e["result"]["prompt_tokens"], e["result"]["prefix_cache_hit_tokens"],
             e["result"]["computed_prefill_tokens"], 1000 * e["timings_seconds"]["prefill"])
            for e in events if e.get("event") == "request_done"
        ]
    else:
        if any(e["snap_mode"] != 0 for e in events):
            raise ValueError("Prompt snapshotting is enabled")
        native = [(e["prompt_tokens"], e["cached_tokens"], e["prompt_tokens"], e["prefill_ms"])
                  for e in events]
    native = native[2:]
    for row, (tokens, cached, computed, ms) in zip(client, native, strict=True):
        if tokens != row["input_tokens"] or cached != 0 or computed != tokens or ms <= 0:
            raise ValueError("Invalid native timings or a cache hit")
        row.update(prefill_ms=ms, prefill_tok_s=1000 * tokens / ms)
    print(json.dumps({"samples": len(client), "median": {
        key: median(row[key] for row in client) for key in ("prefill_ms", "prefill_tok_s", "ttft_ms")
    }, "rows": client}, indent=2))


if __name__ == "__main__":
    main()
