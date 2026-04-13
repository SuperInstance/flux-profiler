"""
FLUX Profiler — measure where FLUX programs spend their cycles.

Profiles:
- Per-opcode execution counts
- Per-instruction timing (simulated + real wall-clock)
- Hot paths (most-executed instruction sequences)
- Register usage patterns
- Memory access patterns
- Instruction-level per-opcode timing
- Function call graph profiling
- Memory allocation tracking per instruction
- Hot-path detection with configurable depth
- Flame graph data generation (JSON format)
- Profile comparison (before/after optimization)
"""
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict


OP_NAMES = {
    0x00:"HALT",0x01:"NOP",0x08:"INC",0x09:"DEC",0x0A:"NOT",0x0B:"NEG",
    0x0C:"PUSH",0x0D:"POP",0x17:"STRIPCONF",0x18:"MOVI",0x19:"ADDI",0x1A:"SUBI",
    0x20:"ADD",0x21:"SUB",0x22:"MUL",0x23:"DIV",0x24:"MOD",
    0x25:"AND",0x26:"OR",0x27:"XOR",
    0x2A:"MIN",0x2B:"MAX",
    0x2C:"CMP_EQ",0x2D:"CMP_LT",0x2E:"CMP_GT",0x2F:"CMP_NE",
    0x3A:"MOV",0x3C:"JZ",0x3D:"JNZ",0x40:"MOVI16",0x43:"JMP",0x46:"LOOP",
    0x48:"CALL",0x49:"RET",
}


@dataclass
class OpcodeProfile:
    opcode: int
    name: str
    count: int
    percentage: float
    cycles_estimate: int = 0  # simulated cycles per op
    total_ns: float = 0.0     # real wall-clock nanoseconds
    avg_ns: float = 0.0       # avg nanoseconds per execution


@dataclass
class HotPath:
    """Sequence of instructions executed most frequently."""
    sequence: List[str]
    count: int
    start_pc: int


@dataclass
class RegisterUsage:
    """How often each register is accessed."""
    register: int
    reads: int
    writes: int
    total: int


@dataclass
class MemoryAllocation:
    """Memory allocation tracking per instruction PC."""
    pc: int
    instruction: str
    push_count: int
    pop_count: int
    net_bytes: int  # positive = stack grew, negative = stack shrunk


@dataclass
class CallGraphNode:
    """A node in the function call graph."""
    caller_pc: int
    callee_pc: int
    call_count: int
    callee_name: str = ""


@dataclass
class InstructionTiming:
    """Per-PC instruction timing data."""
    pc: int
    opcode: int
    name: str
    count: int
    total_ns: float
    avg_ns: float
    min_ns: float
    max_ns: float


@dataclass
class FlameGraphFrame:
    """Single frame in flame graph data."""
    name: str
    value: int  # sample count
    children: List['FlameGraphFrame'] = field(default_factory=list)


class ProfileReport:
    """Complete profiling report for a FLUX program."""
    
    def __init__(
        self,
        total_cycles: int,
        total_instructions: int,
        opcode_profiles: List[OpcodeProfile],
        hot_paths: List[HotPath],
        register_usage: List[RegisterUsage],
        program_size: int,
        ipc: float = 0.0,
        memory_allocations: Optional[List[MemoryAllocation]] = None,
        call_graph: Optional[List[CallGraphNode]] = None,
        instruction_timings: Optional[List[InstructionTiming]] = None,
        flame_graph: Optional[dict] = None,
        labels: Optional[Dict[str, int]] = None,
    ):
        self.total_cycles = total_cycles
        self.total_instructions = total_instructions
        self.opcode_profiles = opcode_profiles
        self.hot_paths = hot_paths
        self.register_usage = register_usage
        self.program_size = program_size
        self.ipc = ipc
        self.memory_allocations = memory_allocations or []
        self.call_graph = call_graph or []
        self.instruction_timings = instruction_timings or []
        self.flame_graph = flame_graph or {}
        self.labels = labels or {}
        
        if self.total_cycles > 0 and self.total_instructions > 0:
            self.ipc = self.total_instructions / self.total_cycles
    
    def to_dict(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "total_instructions": self.total_instructions,
            "program_size": self.program_size,
            "ipc": self.ipc,
            "opcode_profiles": [asdict(op) for op in self.opcode_profiles],
            "hot_paths": [asdict(hp) for hp in self.hot_paths],
            "register_usage": [asdict(ru) for ru in self.register_usage],
            "memory_allocations": [asdict(ma) for ma in self.memory_allocations],
            "call_graph": [asdict(cg) for cg in self.call_graph],
            "instruction_timings": [asdict(it) for it in self.instruction_timings],
            "flame_graph": self.flame_graph,
            "labels": self.labels,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def to_markdown(self) -> str:
        lines = ["# FLUX Profile Report\n"]
        lines.append(f"**Total Cycles:** {self.total_cycles}")
        lines.append(f"**Total Instructions:** {self.total_instructions}")
        lines.append(f"**Program Size:** {self.program_size} bytes")
        lines.append(f"**IPC:** {self.ipc:.2f}\n")
        
        lines.append("## Opcode Distribution\n")
        lines.append("| Opcode | Count | % | Cycles |")
        lines.append("|--------|-------|---|--------|")
        for op in sorted(self.opcode_profiles, key=lambda x: x.count, reverse=True)[:10]:
            cycles = op.cycles_estimate * op.count
            lines.append(f"| {op.name} | {op.count} | {op.percentage:.1f}% | {cycles} |")
        lines.append("")
        
        if self.hot_paths:
            lines.append("## Hot Paths\n")
            for hp in self.hot_paths[:5]:
                lines.append(f"- **PC {hp.start_pc}**: {' → '.join(hp.sequence)} ({hp.count}x)")
            lines.append("")
        
        lines.append("## Register Usage\n")
        for ru in sorted(self.register_usage, key=lambda x: x.total, reverse=True)[:8]:
            lines.append(f"- R{ru.register}: {ru.reads} reads, {ru.writes} writes ({ru.total} total)")
        
        if self.memory_allocations:
            lines.append("\n## Memory Allocations (Top 10)\n")
            lines.append("| PC | Instruction | Pushes | Pops | Net |")
            lines.append("|----|-------------|--------|------|-----|")
            for ma in sorted(self.memory_allocations, key=lambda x: abs(x.net_bytes), reverse=True)[:10]:
                lines.append(f"| {ma.pc} | {ma.instruction} | {ma.push_count} | {ma.pop_count} | {ma.net_bytes:+d} |")
        
        if self.call_graph:
            lines.append("\n## Call Graph\n")
            for cg in self.call_graph[:10]:
                lines.append(f"- PC {cg.caller_pc} → PC {cg.callee_pc} ({cg.call_count}x)")
        
        if self.instruction_timings:
            lines.append("\n## Instruction Timings (Top 10 by total)\n")
            lines.append("| PC | Op | Count | Total (ns) | Avg (ns) |")
            lines.append("|----|-----|-------|------------|----------|")
            for it in sorted(self.instruction_timings, key=lambda x: x.total_ns, reverse=True)[:10]:
                lines.append(f"| {it.pc} | {it.name} | {it.count} | {it.total_ns:.0f} | {it.avg_ns:.1f} |")
        
        if self.labels:
            lines.append("\n## Labels\n")
            for name, pc in sorted(self.labels.items()):
                lines.append(f"- `{name}` → PC {pc}")
        
        return "\n".join(lines)


def compare_profiles(before: ProfileReport, after: ProfileReport) -> dict:
    """Compare two profile reports (before/after optimization)."""
    # Build opcode maps
    before_ops = {op.opcode: op for op in before.opcode_profiles}
    after_ops = {op.opcode: op for op in after.opcode_profiles}
    
    opcode_diffs = []
    all_opcodes = set(list(before_ops.keys()) + list(after_ops.keys()))
    for op in sorted(all_opcodes):
        b = before_ops.get(op)
        a = after_ops.get(op)
        name = (b or a).name
        b_count = b.count if b else 0
        a_count = a.count if a else 0
        count_delta = a_count - b_count
        count_pct = (count_delta / b_count * 100) if b_count > 0 else 0
        
        b_cycles = b.cycles_estimate * b_count if b else 0
        a_cycles = a.cycles_estimate * a_count if a else 0
        cycle_delta = a_cycles - b_cycles
        
        opcode_diffs.append({
            "opcode": op,
            "name": name,
            "before_count": b_count,
            "after_count": a_count,
            "count_delta": count_delta,
            "count_delta_pct": round(count_pct, 1),
            "before_cycles": b_cycles,
            "after_cycles": a_cycles,
            "cycle_delta": cycle_delta,
        })
    
    return {
        "summary": {
            "before_instructions": before.total_instructions,
            "after_instructions": after.total_instructions,
            "instruction_delta": after.total_instructions - before.total_instructions,
            "instruction_delta_pct": round(
                (after.total_instructions - before.total_instructions) / before.total_instructions * 100,
                1,
            ) if before.total_instructions > 0 else 0,
            "before_cycles": before.total_cycles,
            "after_cycles": after.total_cycles,
            "cycle_delta": after.total_cycles - before.total_cycles,
            "cycle_delta_pct": round(
                (after.total_cycles - before.total_cycles) / before.total_cycles * 100,
                1,
            ) if before.total_cycles > 0 else 0,
            "before_ipc": round(before.ipc, 4),
            "after_ipc": round(after.ipc, 4),
            "speedup": round(before.total_cycles / after.total_cycles, 4) if after.total_cycles > 0 else 0,
        },
        "opcode_diffs": opcode_diffs,
    }


class FluxProfiler:
    """Profile FLUX bytecode execution."""
    
    # Estimated cycle costs per opcode
    CYCLE_COSTS = {
        0x00: 1, 0x01: 1, 0x08: 1, 0x09: 1, 0x0A: 1, 0x0B: 1,
        0x0C: 2, 0x0D: 2, 0x18: 1, 0x19: 1, 0x1A: 1,
        0x20: 2, 0x21: 2, 0x22: 3, 0x23: 4, 0x24: 4,
        0x25: 1, 0x26: 1, 0x27: 1,
        0x2C: 2, 0x2D: 2, 0x2E: 2, 0x2F: 2,
        0x3A: 1, 0x3C: 2, 0x3D: 2, 0x40: 1, 0x43: 2, 0x46: 2,
        0x48: 3, 0x49: 3,
    }
    
    def __init__(self, bytecode: List[int], enable_wallclock: bool = False):
        self.bytecode = bytes(bytecode)
        self.opcode_counts: Counter = Counter()
        self.register_reads: Counter = Counter()
        self.register_writes: Counter = Counter()
        self.pc_visits: Counter = Counter()
        self.execution_trace: List[Tuple[int, int]] = []  # (pc, opcode)
        self.total_cycles = 0
        self.enable_wallclock = enable_wallclock
        
        # Wall-clock timing per opcode
        self.opcode_ns: Dict[int, float] = defaultdict(float)
        
        # Per-PC instruction timing
        self.pc_timings: Dict[int, List[float]] = defaultdict(list)
        
        # Memory allocation tracking per PC
        self.pc_push_count: Counter = Counter()
        self.pc_pop_count: Counter = Counter()
        
        # Call graph tracking
        self.call_graph_edges: Counter = Counter()  # (caller_pc, callee_pc) -> count
        self.call_stack: List[int] = []  # PC addresses of callers
        
        # Labels (name -> pc)
        self.labels: Dict[str, int] = {}
        
        # Flame graph call stack samples
        self.flame_samples: List[List[str]] = []
    
    def add_label(self, name: str, pc: int):
        """Add a named label for a PC address."""
        self.labels[name] = pc
    
    def _signed_byte(self, b):
        return b - 256 if b > 127 else b
    
    def _pc_instruction_name(self, pc: int) -> str:
        """Get instruction name at a PC, including label if available."""
        op = self.bytecode[pc] if pc < len(self.bytecode) else 0
        name = OP_NAMES.get(op, f"0x{op:02x}")
        # Find label for this PC
        for label_name, label_pc in self.labels.items():
            if label_pc == pc:
                return f"{name}@{label_name}"
        return name
    
    def profile(self, max_cycles: int = 100000, hot_path_depth: int = 3,
                flame_sample_interval: int = 0) -> ProfileReport:
        """Execute and profile the bytecode.
        
        Args:
            max_cycles: Maximum instructions to execute.
            hot_path_depth: Length of hot path sequences to detect.
            flame_sample_interval: If > 0, collect flame graph samples every N instructions.
        """
        regs = [0] * 64
        stack = [0] * 4096
        sp = 4096
        pc = 0
        halted = False
        total_instructions = 0
        
        while not halted and pc < len(self.bytecode) and total_instructions < max_cycles:
            op = self.bytecode[pc]
            self.opcode_counts[op] += 1
            self.pc_visits[pc] += 1
            self.execution_trace.append((pc, op))
            total_instructions += 1
            
            # Wall-clock timing
            t_start = time.perf_counter_ns() if self.enable_wallclock else 0
            
            # Estimate cycles
            self.total_cycles += self.CYCLE_COSTS.get(op, 1)
            
            # Track register access (simplified)
            if op in (0x08, 0x09, 0x0A, 0x0B, 0x0D):  # 1-reg writes
                if pc+1 < len(self.bytecode):
                    self.register_writes[self.bytecode[pc+1]] += 1
            elif op in (0x0C,):  # PUSH reads
                if pc+1 < len(self.bytecode):
                    self.register_reads[self.bytecode[pc+1]] += 1
            elif op == 0x18:  # MOVI writes
                if pc+1 < len(self.bytecode):
                    self.register_writes[self.bytecode[pc+1]] += 1
            elif op in (0x20, 0x21, 0x22, 0x23, 0x24, 0x2C, 0x2D, 0x2E, 0x2F, 0x3A):
                if pc+3 < len(self.bytecode):
                    self.register_writes[self.bytecode[pc+1]] += 1
                    self.register_reads[self.bytecode[pc+2]] += 1
                    self.register_reads[self.bytecode[pc+3]] += 1
            
            # Memory allocation tracking
            if op == 0x0C:  # PUSH
                self.pc_push_count[pc] += 1
            elif op == 0x0D:  # POP
                self.pc_pop_count[pc] += 1
            
            # Execute
            if op == 0x00: halted = True; pc += 1
            elif op == 0x01: pc += 1
            elif op == 0x08: regs[self.bytecode[pc+1]] += 1; pc += 2
            elif op == 0x09: regs[self.bytecode[pc+1]] -= 1; pc += 2
            elif op == 0x0C: sp -= 1; stack[sp] = regs[self.bytecode[pc+1]]; pc += 2
            elif op == 0x0D: regs[self.bytecode[pc+1]] = stack[sp]; sp += 1; pc += 2
            elif op == 0x18: regs[self.bytecode[pc+1]] = self._signed_byte(self.bytecode[pc+2]); pc += 3
            elif op == 0x19: regs[self.bytecode[pc+1]] += self._signed_byte(self.bytecode[pc+2]); pc += 3
            elif op == 0x20: regs[self.bytecode[pc+1]] = regs[self.bytecode[pc+2]] + regs[self.bytecode[pc+3]]; pc += 4
            elif op == 0x21: regs[self.bytecode[pc+1]] = regs[self.bytecode[pc+2]] - regs[self.bytecode[pc+3]]; pc += 4
            elif op == 0x22: regs[self.bytecode[pc+1]] = regs[self.bytecode[pc+2]] * regs[self.bytecode[pc+3]]; pc += 4
            elif op == 0x23:
                if regs[self.bytecode[pc+3]] != 0:
                    regs[self.bytecode[pc+1]] = regs[self.bytecode[pc+2]] // regs[self.bytecode[pc+3]]
                pc += 4
            elif op == 0x2E: regs[self.bytecode[pc+1]] = 1 if regs[self.bytecode[pc+2]] > regs[self.bytecode[pc+3]] else 0; pc += 4
            elif op == 0x3A: regs[self.bytecode[pc+1]] = regs[self.bytecode[pc+2]]; pc += 4
            elif op == 0x3D:
                if regs[self.bytecode[pc+1]] != 0:
                    pc += self._signed_byte(self.bytecode[pc+2])
                else:
                    pc += 4
            elif op == 0x40:
                imm = self.bytecode[pc+2] | (self.bytecode[pc+3] << 8)
                if imm > 0x7FFF: imm -= 0x10000
                regs[self.bytecode[pc+1]] = imm; pc += 4
            elif op == 0x48:  # CALL
                if pc+1 < len(self.bytecode):
                    callee = self.bytecode[pc+1]
                    self.call_stack.append(pc)
                    self.call_graph_edges[(pc, callee)] += 1
                    pc = callee
            elif op == 0x49:  # RET
                if self.call_stack:
                    ret_pc = self.call_stack.pop()
                    pc = ret_pc + 2  # skip past CALL instruction
                else:
                    pc += 1
            else: pc += 1
            
            # Wall-clock timing end
            if self.enable_wallclock:
                t_end = time.perf_counter_ns()
                elapsed = t_end - t_start
                self.opcode_ns[op] += elapsed
                self.pc_timings[pc - (1 if op in (0x00, 0x01, 0x49) else
                                       2 if op in (0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D) else
                                       3 if op in (0x18, 0x19) else 4)].append(elapsed)
            
            # Flame graph sampling
            if flame_sample_interval > 0 and total_instructions % flame_sample_interval == 0:
                sample = [self._pc_instruction_name(p) for p in reversed(self.call_stack)]
                sample.append(self._pc_instruction_name(
                    self.execution_trace[-1][0] if self.execution_trace else 0
                ))
                self.flame_samples.append(sample)
        
        # Build report
        total_opcodes = sum(self.opcode_counts.values())
        opcode_profiles = [
            OpcodeProfile(
                opcode=op, name=OP_NAMES.get(op, f"0x{op:02x}"),
                count=count, percentage=count/total_opcodes*100 if total_opcodes else 0,
                cycles_estimate=self.CYCLE_COSTS.get(op, 1),
                total_ns=self.opcode_ns.get(op, 0.0),
                avg_ns=self.opcode_ns.get(op, 0.0) / count if count > 0 else 0.0,
            )
            for op, count in self.opcode_counts.most_common()
        ]
        
        # Hot paths: find most common N-instruction sequences
        path_counts: Counter = Counter()
        path_starts: Dict[Tuple, int] = {}
        depth = max(2, min(hot_path_depth, 10))
        for i in range(len(self.execution_trace) - depth + 1):
            seq = tuple(OP_NAMES.get(self.execution_trace[i+j][1], "?") for j in range(depth))
            path_counts[seq] += 1
            if seq not in path_starts:
                path_starts[seq] = self.execution_trace[i][0]
        
        hot_paths = [
            HotPath(sequence=list(seq), count=count, start_pc=path_starts.get(seq, 0))
            for seq, count in path_counts.most_common(5)
        ]
        
        # Register usage
        all_regs = set(list(self.register_reads.keys()) + list(self.register_writes.keys()))
        register_usage = [
            RegisterUsage(
                register=r,
                reads=self.register_reads.get(r, 0),
                writes=self.register_writes.get(r, 0),
                total=self.register_reads.get(r, 0) + self.register_writes.get(r, 0)
            )
            for r in sorted(all_regs)
        ]
        
        # Memory allocations per instruction
        all_mem_pcs = set(list(self.pc_push_count.keys()) + list(self.pc_pop_count.keys()))
        memory_allocations = [
            MemoryAllocation(
                pc=mpc,
                instruction=OP_NAMES.get(self.bytecode[mpc], f"0x{self.bytecode[mpc]:02x}") if mpc < len(self.bytecode) else "???",
                push_count=self.pc_push_count.get(mpc, 0),
                pop_count=self.pc_pop_count.get(mpc, 0),
                net_bytes=self.pc_push_count.get(mpc, 0) - self.pc_pop_count.get(mpc, 0),
            )
            for mpc in sorted(all_mem_pcs)
        ]
        
        # Call graph
        call_graph = [
            CallGraphNode(
                caller_pc=edge[0],
                callee_pc=edge[1],
                call_count=count,
                callee_name=self._pc_instruction_name(edge[1]),
            )
            for edge, count in self.call_graph_edges.most_common()
        ]
        
        # Instruction timings
        instruction_timings = []
        for ipc_pc, timings in self.pc_timings.items():
            if ipc_pc < len(self.bytecode):
                iop = self.bytecode[ipc_pc]
                instruction_timings.append(InstructionTiming(
                    pc=ipc_pc,
                    opcode=iop,
                    name=OP_NAMES.get(iop, f"0x{iop:02x}"),
                    count=len(timings),
                    total_ns=sum(timings),
                    avg_ns=sum(timings) / len(timings) if timings else 0,
                    min_ns=min(timings) if timings else 0,
                    max_ns=max(timings) if timings else 0,
                ))
        instruction_timings.sort(key=lambda x: x.total_ns, reverse=True)
        
        # Flame graph data
        flame_graph = self._build_flame_graph()
        
        return ProfileReport(
            total_cycles=self.total_cycles,
            total_instructions=total_instructions,
            opcode_profiles=opcode_profiles,
            hot_paths=hot_paths,
            register_usage=register_usage,
            program_size=len(self.bytecode),
            memory_allocations=memory_allocations,
            call_graph=call_graph,
            instruction_timings=instruction_timings,
            flame_graph=flame_graph,
            labels=dict(self.labels),
        )
    
    def _build_flame_graph(self) -> dict:
        """Build flame graph data from collected samples."""
        if not self.flame_samples:
            return {"format": "flamegraph", "samples": 0, "roots": []}
        
        # Build a tree from samples
        root: Dict[str, dict] = {"_count": 0, "_children": {}}
        
        for sample in self.flame_samples:
            node = root
            for frame in sample:
                if frame not in node["_children"]:
                    node["_children"][frame] = {"_count": 0, "_children": {}}
                node = node["_children"][frame]
                node["_count"] += 1
            root["_count"] += 1
        
        def _to_tree(node: dict) -> dict:
            children = []
            for name, child in sorted(node["_children"].items(), key=lambda x: -x[1]["_count"]):
                children.append(_to_tree(child))
                children[-1]["name"] = name
            return {
                "value": node["_count"],
                "children": children,
            }
        
        tree = _to_tree(root)
        return {
            "format": "flamegraph",
            "samples": root["_count"],
            "roots": [tree] if tree["value"] > 0 else [],
        }
    
    def get_flamegraph_folded(self) -> str:
        """Get flame graph data in folded stack format (for flamegraph.pl)."""
        lines = []
        for sample in self.flame_samples:
            lines.append(";".join(sample))
        return "\n".join(lines)


# ── Tests ──────────────────────────────────────────────

import unittest


class TestProfiler(unittest.TestCase):
    def test_simple_profile(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        self.assertEqual(report.total_instructions, 2)
        self.assertGreater(report.total_cycles, 0)
    
    def test_opcode_counts(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 0, 0x08, 1, 0x09, 0, 0x00])
        report = p.profile()
        op_names = [o.name for o in report.opcode_profiles]
        self.assertIn("MOVI", op_names)
    
    def test_hot_paths(self):
        # factorial loop: MUL→DEC→JNZ repeats
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        self.assertGreater(len(report.hot_paths), 0)
    
    def test_register_usage(self):
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        report = p.profile()
        regs_used = [r.register for r in report.register_usage]
        self.assertIn(0, regs_used)
    
    def test_ipc(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        self.assertGreater(report.ipc, 0)
    
    def test_markdown_report(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        self.assertIn("Profile Report", md)
        self.assertIn("MOVI", md)
    
    def test_json_report(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        j = report.to_json()
        data = json.loads(j)
        self.assertIn("total_cycles", data)
    
    def test_factorial_profile(self):
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        self.assertGreater(report.total_instructions, 10)
        self.assertGreater(report.total_cycles, 10)
    
    def test_cycle_costs(self):
        # MUL should cost more than ADD
        self.assertGreater(FluxProfiler.CYCLE_COSTS[0x22], FluxProfiler.CYCLE_COSTS[0x20])
    
    def test_empty_program(self):
        p = FluxProfiler([0x00])
        report = p.profile()
        self.assertEqual(report.total_instructions, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
