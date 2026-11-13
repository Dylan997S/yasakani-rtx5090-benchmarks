# RTX 5090 benchmarks for YASAKANI and NInfer DSpine

An independent benchmark of YASAKANI 0.1.0 Speed, YASAKANI 0.1.0 Quality, and NInfer NVFP4 DSpine K11 on one RTX 5090. Snapshot dated November 14, 2026. The tested deployments use different Qwen3.8-27B weight conversions, draft models, and KV formats.

## Results

All speeds are single-stream tokens per second. S, Q, and N denote YASAKANI Speed, YASAKANI Quality, and NInfer DSpine K11.

| Workload | Metric | S | Q | N |
|---|---|---:|---:|---:|
| English prose, 2,048 output tokens | Decode | 277.9 | 235.7 | 160.1 |
| Python, 2,048 output tokens | Decode | 563.6 | 474.4 | 528.8 |
| English short prompts | Engine prefill | 1,873 | 1,753 | 2,959 |
| Python short prompts | Engine prefill | 2,068 | 2,006 | 3,515 |
| 1,716 and 1,724 input tokens, historical comparison | Engine prefill | 3,896.5 | 3,917.2 | 12,959.6 |
| 1,716 and 1,724 input tokens, same-client retest | Engine prefill | 3,809.9 | 3,711.3 | **10,804.9** |
| Same-client retest | HTTP first-token latency | 472.8 ms | 487.9 ms | **174.9 ms** |
| Official long JSONL, 8,192 output tokens | HTTP decode | **1,169.4** | 1,051.3 | 746.0 |
| 120-record shipping JSONL | Decode | 1,113.7 | 1,010.3 | 798.5 |
| Thinking enabled, 24-task quality suite | HTTP decode | **455.1** | 392.1 | 328.6 |
| Thinking enabled, three-task repeat | HTTP decode | **504.1** | 469.5 | 387.8 |

### Zero-cache prefill

In the zero-cache retest, each model processed two prompts of about 1,720 input tokens across six requests, with one output token per request. NInfer DSpine delivered **2.84–2.91×** the median prefill throughput of the two YASAKANI variants (about 10,805 vs. 3,810 and 3,711 tok/s). The same HTTP client measured a median time to first token of about 175 ms on NInfer and 473–488 ms on YASAKANI. Most of that gap appeared during prompt processing. Internal timing boundaries can vary by engine. Kernel-level profiling would be needed to attribute the gap to individual GPU kernels, quantization, or KV-cache handling. [Per-request zero-cache results](data/zero-cache-prefill.json)

YASAKANI's upstream long-JSONL scores were **1,260.0** and **1,159.8** tok/s, measured with a Python wrapper identified as `0.1.0.dev0`. The public `0.1.0` release reached the local values in the table. The public Harness, which reports `0.2.0.dev0`, reached **1,114.1** and **1,028.0** tok/s in local two-request repeats. In a separate same-machine A/B, merging stream events changed Speed from 1,069.9 to 1,069.8 tok/s and Quality from 946.8 to 955.1 tok/s. [Upstream methods](https://github.com/yasakani-dev/yasakani/blob/0205f0624e698693e186cd54430503b6731258ce/docs/MEASUREMENTS.md)

All six local formal long-JSONL responses reached 8,192 tokens and had byte-identical text. All three shipping responses contained 120 parseable records but supplied one label in the `tags` field, which requested two. The NInfer shipping result is from a Pi CLI request with a system message and server-side decode timing.

### Answer quality

| 24 local tasks | S | Q | N |
|---|---:|---:|---:|
| Thinking off, 1,024-token cap | 14/24 | 13/24 | 14/24 |
| Thinking on, 1,024-token cap | 22/24 | 23/24 | 23/24 |
| Replace two selected answers with 4,096-token retests | 23/24 | 24/24 | 23/24 |
| Three-task repeat at 1,024 tokens, thinking on | 3/3 | 3/3 | 3/3 |

The suite contains eight reasoning, six data processing, four format, and six Python tasks. With thinking enabled, strict reasoning scores were 7/8, 8/8, and 8/8. The Speed miss hit the token cap before producing a final answer. In the 4,096-token retest, NInfer still used the entire budget without emitting code for the Python index-selection task. The 24-task totals in that row reuse the other 22 results.

## Reproduce

Use Python 3.12 on Windows and start one inference server at a time. The tested setup used Windows 11 Pro build 26200, driver 616.92, RTX 5090 at a 600 W power limit, Ryzen 7 9800X3D, and 128 GiB RAM. See the [environment and binary hashes](data/environment.json) and the model-pack manifests for [Speed](data/yasakani-speed-sha256.json) and [Quality](data/yasakani-quality-sha256.json).

- Install the engine from the [YASAKANI 0.1.0 release](https://github.com/yasakani-dev/yasakani/releases/tag/v0.1.0). Obtain the two packs listed in the [pinned upstream README](https://github.com/yasakani-dev/yasakani/blob/0205f0624e698693e186cd54430503b6731258ce/README.md) and verify them against the manifests.
- Extract the Windows portable [NInfer v1.2.5-dspine.5 release](https://github.com/Dylan997S/ninfer-dspine-rtx5090/releases/tag/v1.2.5-dspine.5). Use its included conversion instructions and verify the converted model against the snapshot SHA-256. The [pinned conversion source](https://github.com/Dylan997S/ninfer-dspine-rtx5090/tree/2d919071b7e4feaab6486694b994232dcc145074) uses the Tencent DSpine draft.

The example paths below are placeholders. Run the prefill server and the client in separate terminals. Restart with the Quality model directory after measuring Speed. Run each prefill suite against a dedicated server without other requests so the logs can be joined in order.

```powershell
python -X utf8 -u .\yasakani_cold.py serve --model-dir ".\models\speed" --dll ".\engine\yskn.dll" --host 127.0.0.1 --port 18790 --max-ctx 8192 *> speed.log
```

```powershell
python .\bench.py --suite prefill --base http://127.0.0.1:18790/v1 --model qwen3.8-27b --out speed-prefill.jsonl
python .\prefill_metrics.py yasakani speed-prefill.jsonl speed.log
```

For NInfer, from this repository with the portable package and converted model available:

```powershell
$env:NINFER_CUDA_SYNC = 'blocking'
& ".\portable\ninfer-serve.exe" ".\models\qwen27-dspine-nvfp4-tables.v3.ninfer" --spec dspine --draft-tokens 11 --proposal-head full --kv-dtype fp8 --max-context 8192 --context-capacity 8GiB --host-cache-capacity 8GiB --max-concurrency 1 --prefill-chunk 1024 --no-prefix-reuse --host 127.0.0.1 --port 18789 --model-id qwen27-dspine --chat-template ".\portable\tools\chat_templates\qwen3_8.jinja" --request-log-jsonl ninfer-prefill-events.jsonl
```

```powershell
python .\bench.py --suite prefill --base http://127.0.0.1:18789/v1 --model qwen27-dspine --out ninfer-prefill.jsonl
python .\prefill_metrics.py ninfer ninfer-prefill.jsonl ninfer-prefill-events.jsonl
```

Use fresh log and output files for every run. The prefill client sends two warmups before the six measured requests and checks the reported cache hits. The joining script checks the native logs as well. For other suites, use the regular `python -m yasakani serve` entry point with default context and `--max-tokens-cap 16384`. For the matched NInfer long JSONL and quality tests use `--max-context 16384 --prefix-reuse`. Then choose `--suite official-jsonl`, `quality-off`, `quality-on`, `quality-retry`, `quality-short`, or `shipping`. Run `python bench.py --help` for all options. For official JSONL, the client downloads and hash-checks the [pinned upstream fixture](https://github.com/yasakani-dev/yasakani/blob/0205f0624e698693e186cd54430503b6731258ce/bench/20261103/prompts/cases.json); `--upstream-cases` accepts the same file offline. Use the upstream client for an exact replay of its implementation.

Python answers can be inspected and scored in a trusted isolated environment with `--grade-file INPUT.jsonl --execute-reviewed-code` and a new `--out` file. This executes reviewed functions with a restricted namespace and a timeout.

## Dataset and timing

| Suite in [runs.json](data/runs.json) | Source and settings |
|---|---|
| `legacy-native` | Local native-interface YASAKANI and historical NInfer lab build. Two prompts × two rounds per category and engine, 2,048 output tokens, seed 42, 8,192 context, zero cache hits, thinking off. The historical NInfer run used presence penalty 1.5. The client offers HTTP replays of these prompts. |
| `official-jsonl-local` | Fresh local HTTP measurements, two consecutive requests per engine, max 8,192, thinking off, temperature 0. First prompt cold, second prompt reused. |
| `official-jsonl-upstream` | Official upstream records, two requests per YASAKANI pack, separate machine. |
| `jsonl-harness_pair` | Local public Harness using the same DLL and prompt, two consecutive requests per pack. |
| `jsonl-token`, `jsonl-batch`, `jsonl-harness` | Local streaming controls. Two requests per pack and mode, all cold; batched mode coalesces content for about 15.6 ms. |
| `shipping-jsonl` | Local single request per model, temperature 0, thinking off, cap 8,192. NInfer used the Pi frontend. |
| `zero-cache-prefill` | Local cold-prompt HTTP retest, two prompts × three rounds per engine, output limit 1, seed 42, temperature 0, 8,192 context, thinking off. |
| `quality-off`, `quality-on` | 24 requests per model and setting, temperature 0, seed 42, top-p 1, top-k 20, min-p 0, zero requested frequency/presence penalties, cap 1,024 including thinking. |
| `quality-retry`, `quality-short` | Same sampling with thinking enabled. Two prior truncated tasks at 4,096 tokens or three representative tasks at 1,024 tokens. |

HTTP decode uses `(output tokens − 1) / (last chunk time − first chunk time)`, including reasoning chunks when enabled. JSONL scores are the mean of per-request rates. The English, Python and thinking totals pool token intervals and elapsed time. Prefill is the median of input tokens divided by internal prefill time; HTTP first-token latency is measured from request dispatch to the first visible content for the prefill suite. Startup and model load are excluded. Native timings from historical runs can use different internal boundaries.

The data contains selected per-request measurements and quality answers. Model files, binaries and full logs remain local. The custom suite samples single-turn text tasks, including code with small unit-test sets. Agent workflows, multi-request throughput and tool-call quality are outside its coverage.
