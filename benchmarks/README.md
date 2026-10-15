# Benchmarks

`run_benchmark.py` measures real numbers on your machine — dataset throughput,
mock-baseline latency, audio-analysis latency, and voice-synthesis timing when
a backend and voice profile are available. Anything unavailable is reported
as `skipped` with the reason, never invented.

```bash
.venv/Scripts/python benchmarks/run_benchmark.py --out benchmarks/results.json
```

Results are git-ignored (`benchmarks/results.json`, `benchmarks/_tmp/`).
Never paste benchmark numbers into docs or the README without the hardware,
model revision, and commit they were measured with.
