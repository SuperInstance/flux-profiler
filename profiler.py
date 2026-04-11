"""
FLUX Profiler — measure where FLUX programs spend their cycles.

Profiles:
- Per-opcode execution counts
- Per-instruction timing (simulated)
- Hot paths (most-executed instruction sequences)
- Register usage patterns
- Memory access patterns
"""
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import Counter


OP_NAMES = {
    0x00:"HALT",0x01:"NOP",0x08:"INC",0x09:"DEC",0x0A:"NOT",0x0B:"NEG",
    0x0C:"PUSH",0x0D:"POP",0x17:"STRIPCONF",0x18:"MOVI",0x19:"ADDI",0x1A:"SUBI",
    0x20:"ADD",0x21:"SUB",0x22:"MUL",0x23:"DIV",0x24:"MOD",
    0x25:"AND",0x26:"OR",0x27:"XOR",
    0x2A:"MIN",0x2B:"MAX",
    0x2C:"CMP_EQ",0x2D:"CMP_LT",0x2E:"CMP_GT",0x2F:"CMP_NE",
    0x3A:"MOV",0x3C:"JZ",0x3D:"JNZ",0x40:"MOVI16",0x43:"JMP",0x46:"LOOP",
}


@dataclass
class OpcodeProfile:
    opcode: int
    name: str
    count: int
    percentage: float
    cycles_estimate: int = 0  # simulated cycles per op


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
class ProfileReport:
    """Complete profiling report for a FLUX program."""
    total_cycles: int
    total_instructions: int
    opcode_profiles: List[OpcodeProfile]
    hot_paths: List[HotPath]
    register_usage: List[RegisterUsage]
    program_size: int
    ipc: float = 0.0  # instructions per cycle
    
    def __post_init__(self):
        if self.total_cycles > 0 and self.total_instructions > 0:
            self.ipc = self.total_instructions / self.total_cycles
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    def to_markdown(self) -> str:
        lines = ["# FLUX Profile Report\n"]
        lines.append(f"**Total Cycles:** {self.total_cycles}")
        lines.append(f"**Total Instructions:** {self.total_instructions}")
        lines.append(f"**Program Size:** {self.program_size} bytes")
        lines.append(f"**IPC:** {self.ipc:.2f}\n")
        
        lines.append("## Opcode Distribution\n")
        lines.append("| Opcode | Count | % |")
        lines.append("|--------|-------|---|")
        for op in sorted(self.opcode_profiles, key=lambda x: x.count, reverse=True)[:10]:
            lines.append(f"| {op.name} | {op.count} | {op.percentage:.1f}% |")
        lines.append("")
        
        if self.hot_paths:
            lines.append("## Hot Paths\n")
            for hp in self.hot_paths[:5]:
                lines.append(f"- **PC {hp.start_pc}**: {' → '.join(hp.sequence)} ({hp.count}x)")
            lines.append("")
        
        lines.append("## Register Usage\n")
        for ru in sorted(self.register_usage, key=lambda x: x.total, reverse=True)[:8]:
            lines.append(f"- R{ru.register}: {ru.reads} reads, {ru.writes} writes ({ru.total} total)")
        
        return "\n".join(lines)


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
    }
    
    def __init__(self, bytecode: List[int]):
        self.bytecode = bytes(bytecode)
        self.opcode_counts: Counter = Counter()
        self.register_reads: Counter = Counter()
        self.register_writes: Counter = Counter()
        self.pc_visits: Counter = Counter()
        self.execution_trace: List[Tuple[int, int]] = []  # (pc, opcode)
        self.total_cycles = 0
    
    def _signed_byte(self, b):
        return b - 256 if b > 127 else b
    
    def profile(self, max_cycles: int = 100000) -> ProfileReport:
        """Execute and profile the bytecode."""
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
            else: pc += 1
        
        # Build report
        total_opcodes = sum(self.opcode_counts.values())
        opcode_profiles = [
            OpcodeProfile(
                opcode=op, name=OP_NAMES.get(op, f"0x{op:02x}"),
                count=count, percentage=count/total_opcodes*100 if total_opcodes else 0
            )
            for op, count in self.opcode_counts.most_common()
        ]
        
        # Hot paths: find most common 3-instruction sequences
        path_counts: Counter = Counter()
        path_starts: Dict[Tuple, int] = {}
        for i in range(len(self.execution_trace) - 2):
            seq = tuple(OP_NAMES.get(self.execution_trace[i+j][1], "?") for j in range(3))
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
        
        return ProfileReport(
            total_cycles=self.total_cycles,
            total_instructions=total_instructions,
            opcode_profiles=opcode_profiles,
            hot_paths=hot_paths,
            register_usage=register_usage,
            program_size=len(self.bytecode),
        )


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
