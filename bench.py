"""Run the published cases against a locally started OpenAI-compatible server."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
UPSTREAM_CASES = ("https://raw.githubusercontent.com/yasakani-dev/yasakani/"
                  "0205f0624e698693e186cd54430503b6731258ce/bench/20261103/prompts/cases.json")
UPSTREAM_SHA256 = "a4736b56c22788d7a7eb54304afa97027f14d020a6bf19070d397f2fff821c15"
RETRY = {"logic-08", "python-02"}
SHORT = {"logic-06", "format-01", "python-06"}


def workload(suite, upstream_path):
    if suite == "official-jsonl":
        if upstream_path is None:
            with urlopen(UPSTREAM_CASES, timeout=30) as response:
                raw = response.read()
        else:
            raw = upstream_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
            raise ValueError("Upstream cases.json SHA-256 mismatch")
        case, = [c for c in json.loads(raw) if c["name"] == "jsonl_orders_8k"]
        return [{"id": case["name"], "messages": case["messages"]}], case["max_new"]
    if suite.startswith("quality"):
        cases = json.loads((ROOT / "workloads/quality.json").read_text(encoding="utf-8"))
        if suite == "quality-retry":
            cases = [c for c in cases if c["id"] in RETRY]
        if suite == "quality-short":
            cases = [c for c in cases if c["id"] in SHORT]
        return cases, 4096 if suite == "quality-retry" else 1024
    cases = json.loads((ROOT / "workloads/local.json").read_text(encoding="utf-8"))
    group = {"english": "english_prose", "python": "python_code", "prefill": "long_input",
             "shipping": "shipping"}[suite]
    return [c for c in cases if c["group"] == group], {"prefill": 1, "shipping": 8192}.get(suite, 2048)


def stream(url, payload):
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    usage, finish = None, None
    first, last, first_content = None, None, None
    chunks = 0
    parts = []
    start = time.perf_counter()
    buffer = b""
    with urlopen(request, timeout=1800) as response:
        while data := response.read1(65536):
            now_ms = round((time.perf_counter() - start) * 1000, 2)
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        parts.append(content)
                        if first_content is None:
                            first_content = now_ms
                    if content or delta.get("reasoning_content") or delta.get("reasoning"):
                        chunks += 1
                        if first is None:
                            first = now_ms
                        last = now_ms
                    finish = choice.get("finish_reason") or finish
    if usage is None or finish is None or first is None:
        raise RuntimeError("Incomplete SSE response or missing usage")
    elapsed = (last - first) / 1000
    output = usage["completion_tokens"]
    return {
        "input_tokens": usage["prompt_tokens"], "output_tokens": output,
        "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
        "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
        "ttft_ms": first, "first_content_ms": first_content, "decode_seconds": elapsed,
        "decode_tok_s": (output - 1) / elapsed if elapsed > 0 else None,
        "chunks": chunks, "finish": finish, "answer": "".join(parts),
    }


def messages(case):
    if "messages" in case:
        return case["messages"]
    prompt = case["prompt"]
    if "tests" in case:
        prompt += "\nReturn only Python source defining solve. No explanations or Markdown."
    elif case["group"] == "data":
        prompt += "\nReturn only the JSON value. No Markdown."
    return [{"role": "user", "content": prompt}]


def grade(case, answer, execute_code):
    text = answer.strip()
    if "tests" in case:
        if not execute_code:
            return None
        fence = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        try:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-I", str(ROOT / "check_code.py")],
                input=json.dumps([fence[1] if fence else text, case["tests"]]),
                encoding="utf-8", capture_output=True, timeout=4, check=True,
            )
            return all(json.loads(result.stdout))
        except (ValueError, subprocess.SubprocessError):
            return False
    try:
        if case["id"] == "format-01":
            actual = [json.loads(line) for line in text.splitlines()]
        elif case["id"] == "format-04":
            actual = list(csv.reader(text.splitlines(), strict=True))
        elif case["id"] in ("format-02", "format-03"):
            actual = text
        else:
            actual = json.loads(text)
        return actual == case["expected"]
    except (ValueError, csv.Error):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=(
        "official-jsonl", "prefill", "shipping", "english", "python", "quality-off",
        "quality-on", "quality-retry", "quality-short"))
    parser.add_argument("--base", help="Local API base including /v1")
    parser.add_argument("--model", help="API model ID")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--upstream-cases", type=Path, help="Use a locally downloaded upstream cases.json")
    parser.add_argument("--execute-reviewed-code", action="store_true",
                        help="Grade reviewed Python answers from --grade-file")
    parser.add_argument("--grade-file", type=Path, help="Grade saved JSONL instead of running requests")
    args = parser.parse_args()
    if args.execute_reviewed_code and not args.grade_file:
        parser.error("--execute-reviewed-code requires --grade-file")
    cases, max_tokens = workload(args.suite, args.upstream_cases)
    if args.grade_file:
        by_id = {case["id"]: case for case in cases}
        with args.out.open("x", encoding="utf-8") as out:
            for line in args.grade_file.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                row["pass"] = grade(by_id[row["case"]], row["answer"], args.execute_reviewed_code)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if not args.base or not args.model:
        parser.error("--base and --model are required for live runs")
    url = urlsplit(args.base)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("--base must be a loopback HTTP URL")
    repeats = args.repeats if args.repeats is not None else {
        "official-jsonl": 2, "prefill": 3, "english": 2, "python": 2}.get(args.suite, 1)
    if repeats < 1:
        parser.error("--repeats must be positive")
    thinking = args.suite in ("quality-on", "quality-retry", "quality-short")
    base = args.base.rstrip("/") + "/chat/completions"

    def request(items, limit):
        body = {"model": args.model, "messages": items, "max_tokens": limit, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": thinking}}
        if args.suite.startswith("quality"):
            body.update(seed=42, presence_penalty=0, frequency_penalty=0, top_p=1, top_k=20, min_p=0)
        elif args.suite == "prefill":
            body.update(seed=42)
        elif args.suite == "shipping":
            body.update(top_p=0.8, top_k=20, min_p=0)
        return stream(base, body)

    with args.out.open("x", encoding="utf-8") as out:
        if args.suite in ("official-jsonl", "shipping"):
            if args.suite == "official-jsonl":
                for _ in range(2):
                    stream(base, {"model": args.model, "messages": [{"role": "user", "content":
                           "Write two sentences about the sea."}], "max_tokens": 128, "stream": True,
                           "stream_options": {"include_usage": True}, "temperature": 0, "top_p": 0.95,
                           "top_k": 20, "min_p": 0, "chat_template_kwargs": {"enable_thinking": False}})
            for _ in range(2):
                request([{"role": "user", "content": "Write the numbers 1 to 40 as a JSON list."}], 160)
        elif args.suite == "prefill":
            for case in cases:
                request(messages(case), 1)
        # The official single-case test sends consecutive requests for the same prompt.
        for run in range(1, repeats + 1):
            for case in cases:
                row = request(messages(case), max_tokens)
                row.update(case=case["id"], run=run)
                if args.suite == "prefill" and (row["cached_tokens"] != 0 or row["output_tokens"] != 1):
                    raise RuntimeError("Prefill requires zero cache hits and one output token")
                if args.suite.startswith("quality"):
                    row["pass"] = grade(case, row["answer"], False)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(case["id"], run, row["output_tokens"], row["ttft_ms"], row["decode_tok_s"])


if __name__ == "__main__":
    main()
