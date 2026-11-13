"""Start the installed YASAKANI 0.1.0 HTTP server without prompt-state reuse."""

import json
from yasakani.chat import ChatRunner
from yasakani.cli import main


original_run = ChatRunner.run


def run_cold(self, params, emit):
    self.engine.reset()
    self.engine.spec["snap"] = 0
    result = original_run(self, params, emit)
    print(json.dumps({
        "prompt_tokens": result.prompt_tokens,
        "cached_tokens": result.cached_tokens,
        "completion_tokens": result.completion_tokens,
        "prefill_ms": result.prefill_ms,
        "snap_mode": result.snap_mode,
    }), flush=True)
    return result


if __name__ == "__main__":
    ChatRunner.run = run_cold
    raise SystemExit(main())
