"""
FLUX VM Benchmark Suite — comprehensive performance benchmarking for the FLUX bytecode VM.

Measures per-opcode execution timing, standard workload performance, and produces
JSON + Markdown reports suitable for fleet-wide comparison across runtimes.

ISA v2 opcodes: ADD=0x20, SUB=0x21, MUL=0x22, DIV=0x23, PUSH=0x0C, POP=0x0D, JMP=0x43, etc.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any


# ─── ISA v2 Opcode Table ───────────────────────────────────────────────────────

OPCODES: Dict[int, str] = {
    0x00: 'HALT', 0x01: 'NOP',
    0x08: 'INC', 0x09: 'DEC', 0x0A: 'NOT', 0x0B: 'NEG',
    0x0C: 'PUSH', 0x0D: 'POP', 0x0E: 'DUP', 0x0F: 'SWAP',
    0x10: 'LOAD', 0x11: 'STORE',
    0x18: 'MOVI', 0x19: 'ADDI', 0x1A: 'SUBI',
    0x20: 'ADD', 0x21: 'SUB', 0x22: 'MUL', 0x23: 'DIV', 0x24: 'MOD',
    0x25: 'AND', 0x26: 'OR', 0x27: 'XOR',
    0x2A: 'MIN', 0x2B: 'MAX',
    0x2C: 'CMP_EQ', 0x2D: 'CMP_LT', 0x2E: 'CMP_GT', 0x2F: 'CMP_NE',
    0x30: 'SHL', 0x31: 'SHR',
    0x3A: 'MOV', 0x40: 'MOVI16',
    0x43: 'JMP', 0x44: 'JZ', 0x45: 'JNZ', 0x46: 'LOOP',
    0x48: 'CALL', 0x49: 'RET',
    0x4A: 'JLT',  # Jump if flags_lt (less-than)
    0x4B: 'JGT',  # Jump if flags_gt (greater-than)
    0x50: 'SYSCALL',
}

OPCODE_BY_NAME: Dict[str, int] = {v: k for k, v in OPCODES.items()}


# ─── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class MicroResult:
    opcode: int
    name: str
    iterations: int
    total_ns: int
    avg_ns: float
    ops_per_sec: float
    instruction_count: int


@dataclass
class MacroResult:
    name: str
    description: str
    total_ns: int
    instruction_count: int
    expected_result: Optional[int]
    actual_result: Optional[int]
    passed: bool


@dataclass
class BenchmarkResults:
    microbenchmarks: List[MicroResult]
    macrobenchmarks: List[MacroResult]
    timestamp: str
    vm_version: str = "ISA-v2-mini"


# ─── Mini FLUX VM (ISA v2) ─────────────────────────────────────────────────────

class MiniFluxVM:
    """Minimal FLUX VM for benchmarking. 16 registers, stack-based, ISA v2 bytecodes.

    Instruction formats:
      A: opcode only              — HALT, NOP, RET
      B: opcode + reg             — INC r, DEC r, NOT r, NEG r, POP r
      C: opcode + imm8            — PUSH imm8, JMP offset, CALL addr
      D: opcode + reg + imm8      — MOVI r, imm8, ADDI r, imm8, SUBI r, imm8
      E: opcode + reg + reg       — ADD rd rs, SUB rd rs, MOV rd rs, CMP_* rd rs
      Special: JZ/JNZ offset (signed byte, relative)
    """

    def __init__(self) -> None:
        self.registers: List[int] = [0] * 16  # R0-R15
        self.stack: List[int] = []
        self.pc: int = 0
        self.halted: bool = False
        self.instruction_count: int = 0
        self.bytecode: bytearray = bytearray()
        self.flags_eq: bool = False
        self.flags_lt: bool = False
        self.flags_gt: bool = False
        self.call_stack: List[int] = []

    def load(self, bytecode: bytes) -> None:
        self.bytecode = bytearray(bytecode)
        self.pc = 0

    def reset(self) -> None:
        self.registers = [0] * 16
        self.stack = []
        self.pc = 0
        self.halted = False
        self.instruction_count = 0
        self.flags_eq = False
        self.flags_lt = False
        self.flags_gt = False
        self.call_stack = []

    def fetch_byte(self) -> int:
        b = self.bytecode[self.pc]
        self.pc += 1
        return b

    def fetch_signed(self) -> int:
        b = self.fetch_byte()
        return b if b < 128 else b - 256

    def push(self, val: int) -> None:
        self.stack.append(val & 0xFFFFFFFF)

    def pop(self) -> int:
        return self.stack.pop() if self.stack else 0

    def run(self) -> None:
        while not self.halted and self.pc < len(self.bytecode):
            self.step()

    def step(self) -> None:
        opcode = self.fetch_byte()
        self.instruction_count += 1

        if opcode == 0x00:  # HALT
            self.halted = True
        elif opcode == 0x01:  # NOP
            pass
        elif opcode == 0x08:  # INC reg
            r = self.fetch_byte()
            self.registers[r & 0xF] = (self.registers[r & 0xF] + 1) & 0xFFFFFFFF
        elif opcode == 0x09:  # DEC reg
            r = self.fetch_byte()
            self.registers[r & 0xF] = (self.registers[r & 0xF] - 1) & 0xFFFFFFFF
        elif opcode == 0x0A:  # NOT reg
            r = self.fetch_byte()
            self.registers[r & 0xF] = (~self.registers[r & 0xF]) & 0xFFFFFFFF
        elif opcode == 0x0B:  # NEG reg
            r = self.fetch_byte()
            self.registers[r & 0xF] = (-self.registers[r & 0xF]) & 0xFFFFFFFF
        elif opcode == 0x0C:  # PUSH imm8
            val = self.fetch_byte()
            self.push(val)
        elif opcode == 0x0D:  # POP reg
            r = self.fetch_byte()
            self.registers[r & 0xF] = self.pop()
        elif opcode == 0x0E:  # DUP
            if self.stack:
                self.push(self.stack[-1])
        elif opcode == 0x0F:  # SWAP
            if len(self.stack) >= 2:
                self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]
        elif opcode == 0x10:  # LOAD reg
            r = self.fetch_byte()
            self.push(self.registers[r & 0xF])
        elif opcode == 0x11:  # STORE reg
            r = self.fetch_byte()
            self.registers[r & 0xF] = self.pop()
        elif opcode == 0x18:  # MOVI reg, imm8
            r = self.fetch_byte()
            imm = self.fetch_byte()
            self.registers[r & 0xF] = imm
        elif opcode == 0x19:  # ADDI reg, imm8
            r = self.fetch_byte()
            imm = self.fetch_byte()
            self.registers[r & 0xF] = (self.registers[r & 0xF] + imm) & 0xFFFFFFFF
        elif opcode == 0x1A:  # SUBI reg, imm8
            r = self.fetch_byte()
            imm = self.fetch_byte()
            self.registers[r & 0xF] = (self.registers[r & 0xF] - imm) & 0xFFFFFFFF
        elif opcode == 0x20:  # ADD rd, rs (Format E)
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = (self.registers[rd & 0xF] + self.registers[rs & 0xF]) & 0xFFFFFFFF
        elif opcode == 0x21:  # SUB rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = (self.registers[rd & 0xF] - self.registers[rs & 0xF]) & 0xFFFFFFFF
        elif opcode == 0x22:  # MUL rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = (self.registers[rd & 0xF] * self.registers[rs & 0xF]) & 0xFFFFFFFF
        elif opcode == 0x23:  # DIV rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            s = self.registers[rs & 0xF]
            self.registers[rd & 0xF] = (self.registers[rd & 0xF] // s) if s != 0 else 0
        elif opcode == 0x24:  # MOD rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            s = self.registers[rs & 0xF]
            self.registers[rd & 0xF] = (self.registers[rd & 0xF] % s) if s != 0 else 0
        elif opcode == 0x25:  # AND rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = self.registers[rd & 0xF] & self.registers[rs & 0xF]
        elif opcode == 0x26:  # OR rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = self.registers[rd & 0xF] | self.registers[rs & 0xF]
        elif opcode == 0x27:  # XOR rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = self.registers[rd & 0xF] ^ self.registers[rs & 0xF]
        elif opcode == 0x2A:  # MIN rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = min(self.registers[rd & 0xF], self.registers[rs & 0xF])
        elif opcode == 0x2B:  # MAX rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = max(self.registers[rd & 0xF], self.registers[rs & 0xF])
        elif opcode == 0x2C:  # CMP_EQ rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.flags_eq = self.registers[rd & 0xF] == self.registers[rs & 0xF]
            self.flags_lt = self.registers[rd & 0xF] < self.registers[rs & 0xF]
            self.flags_gt = self.registers[rd & 0xF] > self.registers[rs & 0xF]
        elif opcode == 0x2D:  # CMP_LT (same flag logic as CMP_EQ, check lt)
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.flags_eq = self.registers[rd & 0xF] == self.registers[rs & 0xF]
            self.flags_lt = self.registers[rd & 0xF] < self.registers[rs & 0xF]
            self.flags_gt = self.registers[rd & 0xF] > self.registers[rs & 0xF]
        elif opcode == 0x2E:  # CMP_GT
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.flags_eq = self.registers[rd & 0xF] == self.registers[rs & 0xF]
            self.flags_lt = self.registers[rd & 0xF] < self.registers[rs & 0xF]
            self.flags_gt = self.registers[rd & 0xF] > self.registers[rs & 0xF]
        elif opcode == 0x2F:  # CMP_NE
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.flags_eq = self.registers[rd & 0xF] == self.registers[rs & 0xF]
            self.flags_lt = self.registers[rd & 0xF] < self.registers[rs & 0xF]
            self.flags_gt = self.registers[rd & 0xF] > self.registers[rs & 0xF]
        elif opcode == 0x30:  # SHL rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = (self.registers[rd & 0xF] << (self.registers[rs & 0xF] & 0x1F)) & 0xFFFFFFFF
        elif opcode == 0x31:  # SHR rd, rs
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = self.registers[rd & 0xF] >> (self.registers[rs & 0xF] & 0x1F)
        elif opcode == 0x3A:  # MOV rd, rs (Format E)
            rd = self.fetch_byte()
            rs = self.fetch_byte()
            self.registers[rd & 0xF] = self.registers[rs & 0xF]
        elif opcode == 0x40:  # MOVI16 reg, imm16 (Format: opcode reg lo hi)
            r = self.fetch_byte()
            lo = self.fetch_byte()
            hi = self.fetch_byte()
            self.registers[r & 0xF] = lo | (hi << 8)
        elif opcode == 0x43:  # JMP offset (signed relative)
            offset = self.fetch_signed()
            self.pc = (self.pc + offset - 1) & 0xFFFFFFFF  # -1 because pc already advanced past offset
        elif opcode == 0x44:  # JZ offset
            offset = self.fetch_signed()
            if self.flags_eq:
                self.pc = (self.pc + offset - 1) & 0xFFFFFFFF
        elif opcode == 0x45:  # JNZ offset
            offset = self.fetch_signed()
            if not self.flags_eq:
                self.pc = (self.pc + offset - 1) & 0xFFFFFFFF
        elif opcode == 0x46:  # LOOP reg, offset — decrement reg, jump if nonzero
            r = self.fetch_byte()
            offset = self.fetch_signed()
            self.registers[r & 0xF] = (self.registers[r & 0xF] - 1) & 0xFFFFFFFF
            if self.registers[r & 0xF] != 0:
                self.pc = (self.pc + offset - 1) & 0xFFFFFFFF
        elif opcode == 0x48:  # CALL addr
            addr = self.fetch_byte()
            self.call_stack.append(self.pc)
            self.pc = addr
        elif opcode == 0x49:  # RET
            if self.call_stack:
                self.pc = self.call_stack.pop()
        elif opcode == 0x4A:  # JLT offset — jump if flags_lt
            offset = self.fetch_signed()
            if self.flags_lt:
                self.pc = (self.pc + offset - 1) & 0xFFFFFFFF
        elif opcode == 0x4B:  # JGT offset — jump if flags_gt
            offset = self.fetch_signed()
            if self.flags_gt:
                self.pc = (self.pc + offset - 1) & 0xFFFFFFFF
        elif opcode == 0x50:  # SYSCALL (no-op in benchmark VM)
            pass
        else:
            # Unknown opcode — skip
            pass


# ─── Microbenchmark Generation ─────────────────────────────────────────────────

def _generate_microbench(opcode: int, iterations: int) -> bytes:
    """Generate a FLUX bytecode program that exercises an opcode `iterations` times.

    Strategy: For each opcode, generate a tight loop that executes it.
    Uses R0 and R1 as operands, R2 as scratch.
    """
    bc = bytearray()

    if opcode == 0x01:  # NOP
        bc.extend([0x01] * iterations)

    elif opcode == 0x0C:  # PUSH imm8
        for i in range(iterations):
            bc.extend([0x0C, i % 256])

    elif opcode == 0x0D:  # POP reg
        # Push N values first, then pop them
        for i in range(iterations):
            bc.extend([0x0C, i % 256])
        for _ in range(iterations):
            bc.extend([0x0D, 0x00])  # POP R0

    elif opcode == 0x0E:  # DUP
        bc.extend([0x0C, 0x05])  # PUSH 5
        for _ in range(iterations):
            bc.append(0x0E)
        # Drain stack
        for _ in range(iterations + 1):
            bc.extend([0x0D, 0x00])

    elif opcode == 0x0F:  # SWAP
        bc.extend([0x0C, 0x03, 0x0C, 0x07])  # PUSH 3, PUSH 7
        for _ in range(iterations):
            bc.append(0x0F)
        bc.extend([0x0D, 0x00, 0x0D, 0x00])

    elif opcode in (0x08, 0x09, 0x0A, 0x0B):  # INC, DEC, NOT, NEG reg
        bc.extend([0x18, 0x00, 0x01])  # MOVI R0, 1
        for _ in range(iterations):
            bc.extend([opcode, 0x00])  # OP R0

    elif opcode in (0x10,):  # LOAD reg
        bc.extend([0x18, 0x00, 0x2A])  # MOVI R0, 42
        for _ in range(iterations):
            bc.extend([0x10, 0x00])  # LOAD R0 (push to stack)
        for _ in range(iterations):
            bc.extend([0x0D, 0x01])  # POP R1 (drain)

    elif opcode == 0x11:  # STORE reg
        bc.extend([0x18, 0x00, 0x2A])  # MOVI R0, 42
        for _ in range(iterations):
            bc.extend([0x0C, 0x01, 0x11, 0x00])  # PUSH 1, STORE R0

    elif opcode in (0x18, 0x19, 0x1A):  # MOVI, ADDI, SUBI reg, imm8
        for _ in range(iterations):
            bc.extend([opcode, 0x00, 0x01])  # OP R0, 1

    elif opcode in (0x20, 0x21, 0x22, 0x23, 0x24,  # ADD, SUB, MUL, DIV, MOD
                     0x25, 0x26, 0x27,                # AND, OR, XOR
                     0x2A, 0x2B,                      # MIN, MAX
                     0x2C, 0x2D, 0x2E, 0x2F,          # CMP_EQ/LT/GT/NE
                     0x30, 0x31,                      # SHL, SHR
                     0x3A):                           # MOV rd, rs
        # Format E: opcode + rd + rs
        bc.extend([0x18, 0x00, 0x03, 0x18, 0x01, 0x02])  # MOVI R0,3; MOVI R1,2
        for _ in range(iterations):
            bc.extend([opcode, 0x00, 0x01])  # OP R0, R1

    elif opcode == 0x40:  # MOVI16 reg, imm16
        for _ in range(iterations):
            bc.extend([opcode, 0x00, 0x34, 0x12])  # MOVI16 R0, 0x1234

    elif opcode == 0x43:  # JMP (tight loop)
        # Small tight loop: JMP back to self
        loop_size = 3  # JMP + offset
        bc.extend([0x18, 0x02, 0xFF & iterations])  # MOVI R2, iterations
        # Loop body: DEC R2; JNZ back to DEC
        loop_start = len(bc)
        bc.extend([0x09, 0x02])  # DEC R2
        bc.extend([0x0C, 0x00, 0x0D, 0x03])  # PUSH 0, POP R3 (pad)
        jump_back_offset = loop_start - (len(bc) + 1)
        bc.extend([0x45, jump_back_offset & 0xFF])  # JNZ back

    elif opcode in (0x44, 0x45):  # JZ, JNZ
        bc.extend([0x18, 0x02, 0xFF & iterations])  # MOVI R2, iterations
        bc.extend([0x18, 0x03, 0x00])  # MOVI R3, 0
        loop_start = len(bc)
        # Flip R3 each iteration
        bc.extend([0x0C, 0x01, 0x0D, 0x03])  # PUSH 1, POP R3
        bc.extend([0x18, 0x00, 0x18, 0x01])  # CMP for flags: MOVI R0, MOVI R1
        bc.extend([0x2C, 0x00, 0x01])  # CMP_EQ R0, R1 (sets flags_eq=true)
        bc.extend([0x09, 0x02])  # DEC R2
        jump_back_offset = loop_start - (len(bc) + 1)
        bc.extend([opcode, jump_back_offset & 0xFF])  # JZ/JNZ back

    elif opcode == 0x46:  # LOOP reg, offset
        bc.extend([0x18, 0x00, 0xFF & iterations])  # MOVI R0, iterations
        loop_start = len(bc)
        bc.extend([0x0C, 0x00, 0x0D, 0x03])  # PUSH 0, POP R3 (body)
        jump_back_offset = loop_start - (len(bc) + 1)
        bc.extend([0x46, 0x00, jump_back_offset & 0xFF])  # LOOP R0, back

    elif opcode == 0x48:  # CALL
        # Set up a subroutine at address 200 that's a RET
        # Generate enough padding to reach addr 200
        target_addr = 100
        current_len = 2  # we're going to write the CALL first
        # Pad with NOPs
        if len(bc) < target_addr:
            bc.extend([0x01] * (target_addr - len(bc)))
        # Now place the subroutine: PUSH 1, POP R3, RET
        sub_start = len(bc)
        bc.extend([0x0C, 0x01, 0x0D, 0x03, 0x49])
        # Now generate calls
        bc.extend([0x18, 0x02, 0xFF & iterations])  # MOVI R2, iterations
        loop_start = len(bc)
        bc.extend([0x48, sub_start & 0xFF])  # CALL sub
        bc.extend([0x09, 0x02])  # DEC R2
        jump_back_offset = loop_start - (len(bc) + 1)
        bc.extend([0x45, jump_back_offset & 0xFF])  # JNZ back

    elif opcode == 0x49:  # RET
        # RET needs CALL context — benchmark CALL+RET together
        return _generate_microbench(0x48, iterations)

    bc.append(0x00)  # HALT
    return bytes(bc)


# ─── Macro Benchmark Programs ──────────────────────────────────────────────────

def _fibonacci_bytecode(n: int = 20) -> Tuple[bytes, str, int]:
    """Iterative fibonacci(n). Result in R0.

    Algorithm:
      R0 = 0 (fib_prev)
      R1 = 1 (fib_curr)
      R2 = n (counter)
      loop: if R2 == 0, done
        R3 = R0 + R1
        R0 = R1
        R1 = R3
        R2 = R2 - 1
        jmp loop
      done: HALT  (R0 = fib(n-1))
    """
    bc = bytearray()
    bc.extend([0x18, 0x00, 0x00])  # MOVI R0, 0  (fib_prev)
    bc.extend([0x18, 0x01, 0x01])  # MOVI R1, 1  (fib_curr)
    bc.extend([0x18, 0x02, n & 0xFF])  # MOVI R2, n (counter)

    loop_start = len(bc)
    # Check if R2 == 0
    bc.extend([0x18, 0x03, 0x00])  # MOVI R3, 0
    bc.extend([0x2C, 0x02, 0x03])  # CMP_EQ R2, R3
    done_offset_pos = len(bc) + 1  # JZ will be at this position
    bc.extend([0x44, 0x00])  # JZ done (placeholder)

    # R3 = R0 + R1
    bc.extend([0x20, 0x03, 0x00])  # ADD R3, R0
    bc.extend([0x20, 0x03, 0x01])  # ADD R3, R1
    # R0 = R1
    bc.extend([0x3A, 0x00, 0x01])  # MOV R0, R1
    # R1 = R3
    bc.extend([0x3A, 0x01, 0x03])  # MOV R1, R3
    # R2--
    bc.extend([0x09, 0x02])  # DEC R2
    # JMP loop
    jmp_back = loop_start - (len(bc) + 1)
    bc.extend([0x43, jmp_back & 0xFF])  # JMP loop

    # Fix JZ done offset
    done_pos = len(bc)
    bc[done_offset_pos] = (done_pos - done_offset_pos) & 0xFF

    bc.append(0x00)  # HALT — R0 holds result

    expected = _py_fib(n)
    return bytes(bc), f"Iterative fibonacci({n})", expected


def _py_fib(n: int) -> int:
    """Reference fibonacci implementation."""
    if n <= 0:
        return 0
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _sieve_bytecode(limit: int = 50) -> Tuple[bytes, str, int]:
    """Count primes up to limit using trial division. Result in R2.
    Uses countdown approach to avoid complex conditional branching.
    R0 = candidate, R1 = divisor, R2 = prime count, R4 = remaining candidates.
    """
    bc = bytearray()
    remaining = limit - 2  # candidates 2..limit-1
    bc.extend([0x18, 0x00, 0x02])  # MOVI R0, 2 (candidate)
    bc.extend([0x18, 0x02, 0x00])  # MOVI R2, 0 (prime count)
    bc.extend([0x18, 0x04, remaining & 0xFF])  # MOVI R4, remaining

    # outer_loop: countdown R4 from (limit-2) to 0
    outer_start = len(bc)

    # Assume prime: R3 = 1
    bc.extend([0x18, 0x03, 0x01])  # MOVI R3, 1
    bc.extend([0x18, 0x01, 0x02])  # MOVI R1, 2 (divisor start)

    # inner_loop: check if R0 % R1 == 0
    inner_start = len(bc)
    # If R1 == R0, inner done (no divisor found → prime)
    bc.extend([0x2C, 0x01, 0x00])  # CMP_EQ R1, R0
    inner_done_pos = len(bc) + 1
    bc.extend([0x44, 0x00])  # JZ inner_done (if R1==R0, break)

    # Check R0 % R1 == 0
    bc.extend([0x3A, 0x05, 0x00])  # MOV R5, R0
    bc.extend([0x3A, 0x06, 0x01])  # MOV R6, R1
    bc.extend([0x24, 0x05, 0x06])  # MOD R5, R6
    bc.extend([0x18, 0x07, 0x00])  # MOVI R7, 0
    bc.extend([0x2C, 0x05, 0x07])  # CMP_EQ R5, R7
    not_prime_pos = len(bc) + 1
    bc.extend([0x44, 0x00])  # JZ not_prime (if R0%R1==0, not prime)

    # R1 != R0 and R0 % R1 != 0 → continue inner loop
    bc.extend([0x08, 0x01])  # INC R1
    jmp_inner = inner_start - (len(bc) + 1)
    bc.extend([0x43, jmp_inner & 0xFF])  # JMP inner_loop

    # not_prime: R3 = 0, jump to inner_done
    not_prime_label = len(bc)
    bc[not_prime_pos] = (not_prime_label - not_prime_pos) & 0xFF
    bc.extend([0x18, 0x03, 0x00])  # MOVI R3, 0

    # inner_done:
    inner_done_label = len(bc)
    bc[inner_done_pos] = (inner_done_label - inner_done_pos) & 0xFF

    # If R3 == 1 (prime), increment R2
    bc.extend([0x18, 0x07, 0x01])  # MOVI R7, 1
    bc.extend([0x2C, 0x03, 0x07])  # CMP_EQ R3, R7
    skip_inc_pos = len(bc) + 1
    bc.extend([0x45, 0x00])  # JNZ skip_inc (if NOT eq, skip)

    bc.extend([0x08, 0x02])  # INC R2 (found prime)

    # skip_inc:
    skip_inc_label = len(bc)
    bc[skip_inc_pos] = (skip_inc_label - skip_inc_pos) & 0xFF

    # R0++ (next candidate)
    bc.extend([0x08, 0x00])  # INC R0
    # R4-- (countdown)
    bc.extend([0x09, 0x04])  # DEC R4
    bc.extend([0x18, 0x07, 0x00])  # MOVI R7, 0
    bc.extend([0x2C, 0x04, 0x07])  # CMP_EQ R4, R7
    back = outer_start - (len(bc) + 1)
    bc.extend([0x45, back & 0xFF])  # JNZ outer (if R4!=0, continue)

    bc.append(0x00)  # HALT — R2 holds prime count

    expected = _py_count_primes(limit)
    # Move result to R0 for the benchmark runner
    bc_data = bytearray(bc)
    # Insert MOV R0, R2 before HALT: 3 bytes at position len(bc)-1
    bc_data.insert(len(bc)-1, 0x3A)  # MOV
    bc_data.insert(len(bc), 0x00)    # R0
    bc_data.insert(len(bc)+1, 0x02)  # R2
    return bytes(bc_data), f"Trial division prime count (up to {limit})", expected


def _py_count_primes(n: int) -> int:
    """Count primes up to n using trial division."""
    count = 0
    for num in range(2, n):
        is_prime = True
        for div in range(2, num):
            if num % div == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count


def _nested_loops_bytecode(a: int = 50, b: int = 50, c: int = 10) -> Tuple[bytes, str, int]:
    """Triple-nested loop doing ADD. R0 = a * b * c."""
    bc = bytearray()
    bc.extend([0x18, 0x00, 0x00])  # MOVI R0, 0 (accumulator)
    bc.extend([0x18, 0x01, a & 0xFF])  # MOVI R1, a
    bc.extend([0x18, 0x02, b & 0xFF])  # MOVI R2, b
    bc.extend([0x18, 0x03, c & 0xFF])  # MOVI R3, c

    # outer loop (R1)
    outer_start = len(bc)
    bc.extend([0x18, 0x04, b & 0xFF])  # MOVI R4, b (reset mid counter)
    mid_start = len(bc)
    bc.extend([0x18, 0x05, c & 0xFF])  # MOVI R5, c (reset inner counter)
    inner_start = len(bc)
    bc.extend([0x19, 0x00, 0x01])  # ADDI R0, 1
    # LOOP R5, back_to_inner_start (decrements R5, jumps if nonzero)
    # LOOP is 3 bytes (opcode+reg+offset); after fetch pc=len(bc)+3
    # new_pc = pc + offset - 1 => offset = target - len(bc) - 2
    back = inner_start - len(bc) - 2
    bc.extend([0x46, 0x05, back & 0xFF])  # LOOP R5, inner

    # LOOP R4, back_to_mid_start
    back = mid_start - len(bc) - 2
    bc.extend([0x46, 0x04, back & 0xFF])  # LOOP R4, mid

    # LOOP R1, back_to_outer_start
    back = outer_start - len(bc) - 2
    bc.extend([0x46, 0x01, back & 0xFF])  # LOOP R1, outer

    bc.append(0x00)  # HALT — R0 = a * b * c
    return bytes(bc), f"Triple nested loop ({a}x{b}x{c})", a * b * c


def _factorial_bytecode(n: int = 12) -> Tuple[bytes, str, int]:
    """Iterative factorial(n). Result in R0."""
    bc = bytearray()
    bc.extend([0x18, 0x00, 0x01])  # MOVI R0, 1 (result)
    bc.extend([0x18, 0x01, n & 0xFF])  # MOVI R1, n (counter)

    loop_start = len(bc)
    # R0 = R0 * R1
    bc.extend([0x22, 0x00, 0x01])  # MUL R0, R1
    # R1--
    bc.extend([0x09, 0x01])  # DEC R1
    # Check if R1 == 0
    bc.extend([0x18, 0x02, 0x00])  # MOVI R2, 0
    bc.extend([0x2C, 0x01, 0x02])  # CMP_EQ R1, R2
    done_pos = len(bc) + 1
    bc.extend([0x44, 0x00])  # JZ done

    back = loop_start - (len(bc) + 1)
    bc.extend([0x43, back & 0xFF])  # JMP loop

    done_label = len(bc)
    bc[done_pos] = (done_label - done_pos) & 0xFF

    bc.append(0x00)  # HALT — R0 = n!

    # Reference
    import math
    expected = math.factorial(n)
    return bytes(bc), f"Factorial({n})", expected


def _sum_range_bytecode(n: int = 1000) -> Tuple[bytes, str, int]:
    """Sum integers from 1 to n. Result in R0."""
    bc = bytearray()
    bc.extend([0x18, 0x00, 0x00])  # MOVI R0, 0 (sum)
    bc.extend([0x40, 0x01, n & 0xFF, (n >> 8) & 0xFF])  # MOVI16 R1, n

    loop_start = len(bc)
    bc.extend([0x20, 0x00, 0x01])  # ADD R0, R1
    bc.extend([0x09, 0x01])  # DEC R1
    # if R1 != 0, loop
    bc.extend([0x18, 0x02, 0x00])  # MOVI R2, 0
    bc.extend([0x2C, 0x01, 0x02])  # CMP_EQ R1, R2
    back = loop_start - (len(bc) + 1)
    bc.extend([0x45, back & 0xFF])  # JNZ loop (if NOT eq, continue)

    bc.append(0x00)  # HALT
    expected = n * (n + 1) // 2
    return bytes(bc), f"Sum 1..{n}", expected


def _gcd_batch_bytecode() -> Tuple[bytes, str, int]:
    """Compute GCD of 10 pairs using Euclidean algorithm. Sum all GCDs in R0.

    Pairs: (48,18)=6, (100,75)=25, (35,14)=7, (56,49)=7, (91,26)=13,
           (72,16)=8, (85,34)=17, (63,27)=9, (99,44)=11, (120,45)=15
    Expected sum: 6+25+7+7+13+8+17+9+11+15 = 118
    """
    bc = bytearray()
    pairs = [(48,18), (100,75), (35,14), (56,49), (91,26),
             (72,16), (85,34), (63,27), (99,44), (120,45)]

    bc.extend([0x18, 0x00, 0x00])  # MOVI R0, 0 (sum accumulator)

    for a, b in pairs:
        # Set up: R5 = a, R6 = b
        if a <= 255:
            bc.extend([0x18, 0x05, a])  # MOVI R5, a
        else:
            bc.extend([0x40, 0x05, a & 0xFF, (a >> 8) & 0xFF])  # MOVI16 R5, a
        if b <= 255:
            bc.extend([0x18, 0x06, b])  # MOVI R6, b
        else:
            bc.extend([0x40, 0x06, b & 0xFF, (b >> 8) & 0xFF])  # MOVI16 R6, b

        # GCD loop: while R6 != 0: R7 = R5 % R6; R5 = R6; R6 = R7
        gcd_loop = len(bc)
        bc.extend([0x18, 0x07, 0x00])  # MOVI R7, 0
        bc.extend([0x2C, 0x06, 0x07])  # CMP_EQ R6, R7 (R6 == 0?)
        gcd_done_pos = len(bc) + 1
        bc.extend([0x44, 0x00])  # JZ gcd_done (if R6 == 0)

        bc.extend([0x3A, 0x07, 0x05])  # MOV R7, R5
        bc.extend([0x24, 0x07, 0x06])  # MOD R7, R6 (R7 = R5 % R6)
        bc.extend([0x3A, 0x05, 0x06])  # MOV R5, R6
        bc.extend([0x3A, 0x06, 0x07])  # MOV R6, R7
        back = gcd_loop - (len(bc) + 1)
        bc.extend([0x43, back & 0xFF])  # JMP gcd_loop

        gcd_done_label = len(bc)
        bc[gcd_done_pos] = (gcd_done_label - gcd_done_pos) & 0xFF

        # R0 += R5 (add GCD to sum)
        bc.extend([0x20, 0x00, 0x05])  # ADD R0, R5

    bc.append(0x00)  # HALT

    expected = sum(__import__('math').gcd(a, b) for a, b in pairs)
    return bytes(bc), "GCD batch (10 pairs)", expected


MACRO_BENCHMARKS = [
    lambda: _fibonacci_bytecode(20),
    lambda: _sieve_bytecode(30),  # Reduced for bytecode interpreter speed
    lambda: _nested_loops_bytecode(20, 20, 5),
    lambda: _factorial_bytecode(12),
    lambda: _sum_range_bytecode(200),  # Reduced for VM speed
    lambda: _gcd_batch_bytecode(),
]


# ─── Benchmark Runners ─────────────────────────────────────────────────────────

def run_microbenchmarks(iterations: int = 10000) -> List[MicroResult]:
    """Run per-opcode microbenchmarks and return timing results."""
    results: List[MicroResult] = []

    # Opcodes to benchmark (skip HALT and SYSCALL)
    benchmark_opcodes = [
        (0x01, 'NOP'),
        (0x08, 'INC'), (0x09, 'DEC'), (0x0A, 'NOT'), (0x0B, 'NEG'),
        (0x0C, 'PUSH'), (0x0D, 'POP'), (0x0E, 'DUP'), (0x0F, 'SWAP'),
        (0x10, 'LOAD'), (0x11, 'STORE'),
        (0x18, 'MOVI'), (0x19, 'ADDI'), (0x1A, 'SUBI'),
        (0x20, 'ADD'), (0x21, 'SUB'), (0x22, 'MUL'), (0x23, 'DIV'), (0x24, 'MOD'),
        (0x25, 'AND'), (0x26, 'OR'), (0x27, 'XOR'),
        (0x2A, 'MIN'), (0x2B, 'MAX'),
        (0x2C, 'CMP_EQ'), (0x2D, 'CMP_LT'), (0x2E, 'CMP_GT'), (0x2F, 'CMP_NE'),
        (0x30, 'SHL'), (0x31, 'SHR'),
        (0x3A, 'MOV'), (0x40, 'MOVI16'),
        (0x43, 'JMP'), (0x44, 'JZ'), (0x45, 'JNZ'), (0x46, 'LOOP'),
        (0x48, 'CALL'), (0x49, 'RET'),
    ]

    for opcode_val, opcode_name in benchmark_opcodes:
        try:
            program = _generate_microbench(opcode_val, iterations)
            vm = MiniFluxVM()
            vm.load(program)

            start = time.perf_counter_ns()
            vm.run()
            elapsed = time.perf_counter_ns() - start

            avg_ns = elapsed / max(vm.instruction_count, 1)
            ops_per_sec = (vm.instruction_count / elapsed * 1e9) if elapsed > 0 else 0

            results.append(MicroResult(
                opcode=opcode_val,
                name=opcode_name,
                iterations=iterations,
                total_ns=elapsed,
                avg_ns=round(avg_ns, 2),
                ops_per_sec=round(ops_per_sec, 0),
                instruction_count=vm.instruction_count,
            ))
        except Exception as e:
            results.append(MicroResult(
                opcode=opcode_val,
                name=opcode_name,
                iterations=iterations,
                total_ns=0,
                avg_ns=0,
                ops_per_sec=0,
                instruction_count=0,
            ))

    return results


def run_macrobenchmarks() -> List[MacroResult]:
    """Run standard workload macro benchmarks."""
    results: List[MacroResult] = []

    for bench_fn in MACRO_BENCHMARKS:
        try:
            program, description, expected = bench_fn()
            vm = MiniFluxVM()
            vm.load(program)

            start = time.perf_counter_ns()
            vm.run()
            elapsed = time.perf_counter_ns() - start

            # Get result from R0
            actual = vm.registers[0] & 0xFFFFFFFF

            # Handle factorial overflow in 32-bit
            if "Factorial" in description:
                actual_raw = actual
                # For large factorials, check modulo 2^32
                import math
                if expected > 0xFFFFFFFF:
                    expected = expected % (2**32)
                # Recompute factorial in modular arithmetic
                expected_mod = 1
                n = 15
                for i in range(1, n + 1):
                    expected_mod = (expected_mod * i) % (2**32)
                expected = expected_mod

            passed = actual == expected

            results.append(MacroResult(
                name=description,
                description=description,
                total_ns=elapsed,
                instruction_count=vm.instruction_count,
                expected_result=expected,
                actual_result=actual,
                passed=passed,
            ))
        except Exception as e:
            results.append(MacroResult(
                name="ERROR",
                description=str(e),
                total_ns=0,
                instruction_count=0,
                expected_result=None,
                actual_result=None,
                passed=False,
            ))

    return results


# ─── Output Formatting ─────────────────────────────────────────────────────────

def run_all() -> BenchmarkResults:
    """Run all benchmarks and return structured results."""
    micro = run_microbenchmarks()
    macro = run_macrobenchmarks()
    return BenchmarkResults(
        microbenchmarks=micro,
        macrobenchmarks=macro,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S UTC", time.gmtime()),
    )


def to_json(results: BenchmarkResults) -> str:
    """Convert results to machine-readable JSON."""
    return json.dumps(asdict(results), indent=2)


def to_markdown(results: BenchmarkResults) -> str:
    """Convert results to human-readable Markdown table."""
    lines = [
        "# FLUX VM Benchmark Results",
        "",
        f"**Timestamp:** {results.timestamp}",
        f"**VM Version:** {results.vm_version}",
        "",
        "## Microbenchmarks — Per-Opcode Performance",
        "",
        "| Opcode | Name | Instructions | Total (ns) | Avg (ns/op) | Ops/sec |",
        "|--------|------|-------------|------------|-------------|---------|",
    ]

    for m in results.microbenchmarks:
        lines.append(
            f"| 0x{m.opcode:02X} | {m.name:8s} | {m.instruction_count:>11,} | "
            f"{m.total_ns:>10,} | {m.avg_ns:>11.2f} | {m.ops_per_sec:>12,.0f} |"
        )

    # Summary stats
    valid = [m for m in results.microbenchmarks if m.total_ns > 0]
    if valid:
        fastest = min(valid, key=lambda x: x.avg_ns)
        slowest = max(valid, key=lambda x: x.avg_ns)
        avg_all = sum(m.avg_ns for m in valid) / len(valid)

        lines.extend([
            "",
            f"**Fastest opcode:** {fastest.name} ({fastest.avg_ns:.2f} ns/op)",
            f"**Slowest opcode:** {slowest.name} ({slowest.avg_ns:.2f} ns/op)",
            f"**Average:** {avg_all:.2f} ns/op",
            "",
        ])

    lines.extend([
        "## Macro Benchmarks — Standard Workloads",
        "",
        "| Benchmark | Instructions | Time (ns) | Expected | Actual | Status |",
        "|-----------|-------------|-----------|----------|--------|--------|",
    ])

    for m in results.macrobenchmarks:
        status = "PASS" if m.passed else "FAIL"
        exp_str = str(m.expected_result) if m.expected_result is not None else "N/A"
        act_str = str(m.actual_result) if m.actual_result is not None else "N/A"
        lines.append(
            f"| {m.name:40s} | {m.instruction_count:>11,} | {m.total_ns:>9,} | "
            f"{exp_str:>8s} | {act_str:>6s} | {status:4s} |"
        )

    macro_pass = sum(1 for m in results.macrobenchmarks if m.passed)
    macro_total = len(results.macrobenchmarks)
    lines.extend([
        "",
        f"**Macro benchmark results:** {macro_pass}/{macro_total} passed",
        "",
    ])

    return "\n".join(lines)


def save_results(filepath: str, results: BenchmarkResults) -> None:
    """Save benchmark results to both JSON and Markdown files."""
    json_path = filepath + ".json"
    md_path = filepath + ".md"

    with open(json_path, "w") as f:
        f.write(to_json(results))

    with open(md_path, "w") as f:
        f.write(to_markdown(results))


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  FLUX VM Benchmark Suite — ISA v2")
    print("=" * 72)
    print()

    results = run_all()

    print(to_markdown(results))

    output_base = "/home/z/my-project/download/benchmark-results-2026-04-13"
    save_results(output_base, results)
    print(f"\nResults saved to:")
    print(f"  {output_base}.json")
    print(f"  {output_base}.md")
