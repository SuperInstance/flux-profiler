# flux-profiler

> Bytecode profiler measuring per-opcode counts, hot paths, register usage, and cycle estimation for FLUX programs.

## What This Is

`flux-profiler` is a Python module that **profiles FLUX bytecode execution** — it runs a program and produces a `ProfileReport` with opcode distribution, hot instruction sequences, register read/write patterns, IPC (instructions per cycle), and both JSON and markdown output formats.

## Role in the FLUX Ecosystem

Performance analysis is critical for optimizing agent bytecode:

- **`flux-timeline`** shows *when* things happen; profiler shows *how often*
- **`flux-signatures`** estimates complexity statically; profiler measures it dynamically
- **`flux-coverage`** tracks which instructions ran; profiler counts *how many times*
- **`flux-decompiler`** helps understand code structure; profiler finds the bottlenecks
- **`flux-debugger`** helps find bugs; profiler helps find slowness

## Key Features

| Feature | Description |
|---------|-------------|
| **Opcode Distribution** | Count and percentage for every opcode executed |
| **Hot Path Detection** | Top 5 most-executed 3-instruction sequences |
| **Register Usage** | Read/write counts per register |
| **Cycle Estimation** | Per-opcode cost model (MUL=3, DIV=4, ADD=2, etc.) |
| **IPC Metric** | Instructions per cycle ratio |
| **JSON Export** | Machine-readable `to_json()` for pipeline integration |
| **Markdown Reports** | Human-readable `to_markdown()` for code review |
| **30+ Opcodes** | Full FLUX ISA support including logical and comparison ops |

## Quick Start

```python
from flux_profiler import FluxProfiler

# Profile a factorial program
bytecode = [0x18, 0, 10, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, -6, 0, 0x00]
profiler = FluxProfiler(bytecode)
report = profiler.profile()

print(f"Total cycles: {report.total_cycles}")
print(f"Total instructions: {report.total_instructions}")
print(f"IPC: {report.ipc:.2f}")

# Top opcodes
for op in report.opcode_profiles[:5]:
    print(f"  {op.name}: {op.count} ({op.percentage:.1f}%)")

# Hot paths
for hp in report.hot_paths[:3]:
    print(f"  PC {hp.start_pc}: {' -> '.join(hp.sequence)} ({hp.count}x)")

# Export
print(report.to_markdown())
print(report.to_json())
```

## Running Tests

```bash
python -m pytest tests/ -v
# or
python profiler.py
```

## Related Fleet Repos

- [`flux-timeline`](https://github.com/SuperInstance/flux-timeline) — Execution tracing
- [`flux-signatures`](https://github.com/SuperInstance/flux-signatures) — Static pattern analysis
- [`flux-coverage`](https://github.com/SuperInstance/flux-coverage) — Code coverage
- [`flux-decompiler`](https://github.com/SuperInstance/flux-decompiler) — Bytecode decompilation
- [`flux-debugger`](https://github.com/SuperInstance/flux-debugger) — Step debugger

## License

Part of the [SuperInstance](https://github.com/SuperInstance) FLUX fleet.
