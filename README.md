# FLUX Profiler — Performance Profiler

Measure where FLUX programs spend their cycles.

## Features
- **Opcode counts**: How often each instruction executes
- **Cycle estimates**: Per-opcode simulated costs (MUL=3, DIV=4, ADD=2, etc.)
- **Hot paths**: Most-executed 3-instruction sequences
- **Register usage**: Read/write counts per register
- **Reports**: Markdown and JSON output

## Usage

```python
from profiler import FluxProfiler

p = FluxProfiler([0x18, 0, 10, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
report = p.profile()
print(report.to_markdown())
print(f"IPC: {report.ipc:.2f}")
```

10 tests passing.
