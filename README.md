# FLUX Profiler

Performance profiler and benchmarking suite for the FLUX VM (ISA v2).

## Overview

`flux-profiler` provides standardized benchmarking infrastructure for measuring and comparing FLUX VM implementations. It includes:

- **20 standard benchmarks** covering arithmetic, control flow, memory, stack, floating point, and A2A operations
- **Embedded MiniVM** — a minimal ISA v2 interpreter for portable benchmark execution
- **Statistical analysis** — mean/median/stddev, throughput, efficiency scoring, significance testing
- **Report generation** — Markdown, JSON, and CSV output formats
- **Cross-VM comparison** — rank VMs, compute speedup ratios, identify bottlenecks

## Quick Start

```python
from flux_profiler import StandardBenchmarks, Profiler, MiniVMAdapter

# Set up profiler with MiniVM
profiler = Profiler()
profiler.register_vm(MiniVMAdapter())

# Run all benchmarks
benchmarks = StandardBenchmarks.all()
results = profiler.run_suite(benchmarks, "MiniVM-Python", iterations=10)

# Generate report
from flux_profiler import BenchmarkReporter
print(BenchmarkReporter.to_markdown(results))
```

## Installation

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Architecture

- `benchmarks.py` — ISA v2 bytecode benchmark definitions
- `profiler.py` — Benchmark execution orchestration
- `stats.py` — Statistical analysis and comparison
- `reporter.py` — Report generation (MD/JSON/CSV)
- `vm_adapter.py` — VM adapter interface + embedded MiniVM
