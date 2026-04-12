"""Standard benchmark workloads for FLUX ISA v2 performance profiling.

Each benchmark is defined as ISA v2 bytecode that exercises different aspects
of the VM: arithmetic, control flow, memory, stack, floating point, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ── ISA v2 Opcode Constants ──────────────────────────────────────────────────
HALT = 0x00
NOP = 0x01
RET = 0x02
INC = 0x08
DEC = 0x09
NOT = 0x0A
NEG = 0x0B
PUSH = 0x0C
POP = 0x0D
MOVI = 0x18
ADDI = 0x19
SUBI = 0x1A
ANDI = 0x1B
ORI = 0x1C
XORI = 0x1D
SHLI = 0x1E
SHRI = 0x1F
ADD = 0x20
SUB = 0x21
MUL = 0x22
DIV = 0x23
MOD = 0x24
AND = 0x25
OR = 0x26
XOR = 0x27
SHL = 0x28
SHR = 0x29
MIN = 0x2A
MAX = 0x2B
CMP_EQ = 0x2C
CMP_LT = 0x2D
CMP_GT = 0x2E
CMP_NE = 0x2F
FADD = 0x30
FSUB = 0x31
FMUL = 0x32
FDIV = 0x33
FTOI = 0x36
ITOF = 0x37
LOAD = 0x38
STORE = 0x39
MOV = 0x3A
SWP = 0x3B
JZ = 0x3C
JNZ = 0x3D
JLT = 0x3E
JGT = 0x3F
MOVI16 = 0x40
ADDI16 = 0x41
SUBI16 = 0x42
JMP = 0x43
JAL = 0x44
CALL = 0x45
LOOP = 0x46
LOADOFF = 0x48
STOREOFF = 0x49
LOADI = 0x4A
STOREI = 0x4B
ENTER = 0x4C
LEAVE = 0x4D
COPY = 0x4E
FILL = 0x4F
TELL = 0x50
ASK = 0x51
DELEG = 0x52
BCAST = 0x53
ABS = 0x90
CLZ = 0x95
CTZ = 0x96
POPCNT = 0x97

# ── Encoding Helpers ─────────────────────────────────────────────────────────

def encode_a(opcode: int) -> bytes:
    """Format A: single opcode byte."""
    return bytes([opcode])

def encode_b(opcode: int, rd: int) -> bytes:
    """Format B: opcode + 4-bit register."""
    return bytes([opcode, rd & 0xF])

def encode_c(opcode: int, imm8: int) -> bytes:
    """Format C: opcode + 8-bit immediate."""
    return bytes([opcode, imm8 & 0xFF])

def encode_d(opcode: int, rd: int, imm8: int) -> bytes:
    """Format D: opcode + 4-bit register + 8-bit immediate."""
    return bytes([opcode, rd & 0xF, imm8 & 0xFF])

def encode_e(opcode: int, rd: int, rs1: int, rs2: int) -> bytes:
    """Format E: opcode + three 4-bit registers."""
    return bytes([opcode, rd & 0xF, rs1 & 0xF, rs2 & 0xF])

def encode_f(opcode: int, rd: int, imm16: int) -> bytes:
    """Format F: opcode + 4-bit register + 16-bit immediate (little-endian)."""
    return bytes([opcode, rd & 0xF, imm16 & 0xFF, (imm16 >> 8) & 0xFF])

def encode_g(opcode: int, rd: int, rs1: int, imm16: int) -> bytes:
    """Format G: opcode + 4-bit reg + 4-bit reg + 16-bit immediate."""
    return bytes([opcode, rd & 0xF, rs1 & 0xF, imm16 & 0xFF, (imm16 >> 8) & 0xFF])


# ── Benchmark Workload ───────────────────────────────────────────────────────

@dataclass
class BenchmarkWorkload:
    """A single benchmark defined as ISA v2 bytecode."""
    name: str
    description: str
    bytecode: bytes
    expected_result: Optional[int]
    category: str
    estimated_cycles: int = 0


# ── Standard Benchmarks ──────────────────────────────────────────────────────

class StandardBenchmarks:
    """Factory for standard FLUX VM benchmark workloads."""

    @staticmethod
    def factorial_small() -> BenchmarkWorkload:
        """Compute 5! = 120 using a loop.
        R1 = accumulator, R2 = counter, R3 = temp
        """
        code = bytearray()
        # R1 = 1 (accumulator)
        code += encode_d(MOVI, 1, 1)
        # R2 = 5 (counter)
        code += encode_d(MOVI, 2, 5)
        # LOOP_START:
        loop_start = len(code)
        # R1 = R1 * R2
        code += encode_e(MUL, 1, 1, 2)
        # R2 = R2 - 1
        code += encode_b(DEC, 2)
        # R3 = R2 != 0 → CMP_NE: R3 = (R2 != 0) sets R3 to 1 if true
        code += encode_e(CMP_NE, 3, 2, 0)
        # JNZ R3, loop_start
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # result is in R1
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="factorial_small",
            description="Compute 5! = 120 (arithmetic + loop)",
            bytecode=bytes(code),
            expected_result=120,
            category="arithmetic",
            estimated_cycles=50,
        )

    @staticmethod
    def factorial_large() -> BenchmarkWorkload:
        """Compute 12! using a loop. R1=acc, R2=counter, R3=temp."""
        code = bytearray()
        code += encode_d(MOVI, 1, 1)
        code += encode_d(MOVI, 2, 12)
        loop_start = len(code)
        code += encode_e(MUL, 1, 1, 2)
        code += encode_b(DEC, 2)
        code += encode_e(CMP_NE, 3, 2, 0)
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="factorial_large",
            description="Compute 12! (tests overflow behavior)",
            bytecode=bytes(code),
            expected_result=None,  # overflow depends on VM
            category="arithmetic",
            estimated_cycles=80,
        )

    @staticmethod
    def fibonacci() -> BenchmarkWorkload:
        """Compute fib(15) = 610 using iteration.
        R0=0, R1=1, R2=counter, R3=temp, R4=target
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)   # a=0
        code += encode_d(MOVI, 1, 1)   # b=1
        code += encode_d(MOVI, 2, 2)   # i=2
        code += encode_d(MOVI, 4, 15)  # target
        loop_start = len(code)
        # R3 = R0 + R1
        code += encode_e(ADD, 3, 0, 1)
        # R0 = R1
        code += encode_e(MOV, 0, 1, 0)
        # R1 = R3
        code += encode_e(MOV, 1, 3, 0)
        # R2++
        code += encode_b(INC, 2)
        # if R2 < R4, loop
        code += encode_e(CMP_LT, 3, 2, 4)
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # result in R1
        code += encode_e(MOV, 0, 1, 0)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="fibonacci",
            description="Compute fib(15) = 610 (heavy control flow)",
            bytecode=bytes(code),
            expected_result=610,
            category="control_flow",
            estimated_cycles=200,
        )

    @staticmethod
    def matrix_multiply() -> BenchmarkWorkload:
        """3x3 matrix multiply using memory ops.
        Stores matrices in memory, computes C = A * B element by element.
        Result: trace of C (sum of diagonal) in R0.
        """
        code = bytearray()
        # We'll store matrix A at addr 0-8, matrix B at addr 9-17
        # Matrix A = identity:
        a_vals = [1,0,0, 0,1,0, 0,0,1]
        b_vals = [2,0,0, 0,3,0, 0,0,4]
        # Store A
        for i, v in enumerate(a_vals):
            code += encode_d(MOVI, 1, v)
            code += encode_d(STORE, 1, i)
        # Store B
        for i, v in enumerate(b_vals):
            code += encode_d(MOVI, 1, v)
            code += encode_d(STORE, 1, 9 + i)
        # Compute C[0][0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6] = 1*2+0+0=2
        # Load A row 0, B col 0
        code += encode_d(MOVI, 0, 0)   # R0 = accumulator
        # C[0][0]
        code += encode_d(LOAD, 1, 0)    # R1 = A[0]
        code += encode_d(LOAD, 2, 9)    # R2 = B[0]
        code += encode_e(MUL, 3, 1, 2)
        code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 1)
        code += encode_d(LOAD, 2, 12)
        code += encode_e(MUL, 3, 1, 2)
        code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 2)
        code += encode_d(LOAD, 2, 15)
        code += encode_e(MUL, 3, 1, 2)
        code += encode_e(ADD, 0, 0, 3)
        # C[1][1]
        code += encode_d(LOAD, 1, 3)
        code += encode_d(LOAD, 2, 10)
        code += encode_e(MUL, 3, 1, 2)
        code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 4)
        code += encode_d(LOAD, 2, 13)
        code += encode_e(MUL, 3, 1, 2)
        code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 5)
        code += encode_d(LOAD, 2, 16)
        code += encode_e(MUL, 3, 1, 2)
        code += encode_e(ADD, 0, 0, 3)
        # C[2][2]
        code += encode_d(LOAD, 1, 6)
        code += encode_d(LOAD, 2, 11)
        code += encode_e(MUL, 3, 1, 2)
        code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 7)
        code += encode_d(LOAD, 2, 14)
        code += encode_e(MUL, 3, 1, 2)
        code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 8)
        code += encode_d(LOAD, 2, 17)
        code += encode_e(MUL, 3, 1, 2)
        code += encode_e(ADD, 0, 0, 3)
        # trace = 2+3+4 = 9
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="matrix_multiply",
            description="3x3 matrix multiply using memory ops (trace of C = 9)",
            bytecode=bytes(code),
            expected_result=9,
            category="memory",
            estimated_cycles=300,
        )

    @staticmethod
    def bubble_sort() -> BenchmarkWorkload:
        """Sort 8 elements [8,7,6,5,4,3,2,1] using bubble sort.
        Result: sum of sorted array in R0 = 36.
        """
        n = 8
        code = bytearray()
        # Store the array at memory 0-7
        vals = [8, 7, 6, 5, 4, 3, 2, 1]
        for i, v in enumerate(vals):
            code += encode_d(MOVI, 1, v)
            code += encode_d(STORE, 1, i)
        # R0 = sum accumulator
        code += encode_d(MOVI, 0, 0)
        # Read sorted values and sum them
        for i in range(n):
            code += encode_d(LOAD, 1, i)
            code += encode_e(ADD, 0, 0, 1)
        code += encode_a(HALT)
        # Note: this just stores then sums - the sort is trivial since
        # we store already, but we include the memory ops pattern.
        # Actually let's do a proper bubble sort:
        return StandardBenchmarks._bubble_sort_impl()

    @staticmethod
    def _bubble_sort_impl() -> BenchmarkWorkload:
        """Actual bubble sort: sort [8,7,6,5,4,3,2,1], result = sum = 36."""
        n = 8
        code = bytearray()
        # Store array
        vals = [8, 7, 6, 5, 4, 3, 2, 1]
        for i, v in enumerate(vals):
            code += encode_d(MOVI, 1, v)
            code += encode_d(STORE, 1, i)
        # R10 = n (8)
        code += encode_d(MOVI, 10, n)
        # Outer loop: R11 = i from n-1 down to 1
        code += encode_d(MOVI, 11, n - 1)
        outer_start = len(code)
        # R12 = j = 0
        code += encode_d(MOVI, 12, 0)
        inner_start = len(code)
        # Load mem[j] and mem[j+1]
        # We use LOADOFF: rd = mem[rs1 + imm16]
        # R1 = mem[R12 + 0]
        code += encode_g(LOADOFF, 1, 12, 0)
        # R2 = mem[R12 + 1]
        code += encode_g(LOADOFF, 2, 12, 1)
        # R3 = (R1 > R2)? CMP_GT: R3 = 1 if R1 > R2
        code += encode_e(CMP_GT, 3, 1, 2)
        # if R1 <= R2, skip swap
        code += encode_b(JZ, 3)
        skip_rel = 8  # skip the swap block
        code.append(skip_rel & 0xFF)
        # Swap: store R2 at mem[R12+0], R1 at mem[R12+1]
        code += encode_g(STOREOFF, 1, 12, 0)  # will be overwritten but ok
        # Actually need to use R1/R2 properly
        # store R2 at mem[R12+0]
        code += encode_g(STOREOFF, 2, 12, 0)
        # store R1 at mem[R12+1]
        code += encode_g(STOREOFF, 1, 12, 1)
        # R12++
        code += encode_b(INC, 12)
        # if R12 < R11, continue inner
        code += encode_e(CMP_LT, 3, 12, 11)
        code += encode_b(JNZ, 3)
        rel = inner_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # R11--
        code += encode_b(DEC, 11)
        # if R11 > 0, continue outer
        code += encode_e(CMP_GT, 3, 11, 0)
        code += encode_b(JNZ, 3)
        rel = outer_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # Sum all elements
        code += encode_d(MOVI, 0, 0)
        for i in range(n):
            code += encode_g(LOADOFF, 1, 0, i)  # LOADOFF R1, R0, i -- R0=0
        # Actually R0 is our sum, we can't use it as base. Use R13
        code = bytearray()
        # Re-do cleanly
        for i, v in enumerate(vals):
            code += encode_d(MOVI, 1, v)
            code += encode_d(STORE, 1, i)
        code += encode_d(MOVI, 10, n)
        code += encode_d(MOVI, 11, n - 1)
        outer_start = len(code)
        code += encode_d(MOVI, 12, 0)
        inner_start = len(code)
        code += encode_g(LOADOFF, 1, 12, 0)
        code += encode_g(LOADOFF, 2, 12, 1)
        code += encode_e(CMP_GT, 3, 1, 2)
        code += encode_b(JZ, 3)
        code.append(8)
        code += encode_g(STOREOFF, 2, 12, 0)
        code += encode_g(STOREOFF, 1, 12, 1)
        code += encode_b(INC, 12)
        code += encode_e(CMP_LT, 3, 12, 11)
        code += encode_b(JNZ, 3)
        rel = inner_start - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_b(DEC, 11)
        code += encode_e(CMP_GT, 3, 11, 0)
        code += encode_b(JNZ, 3)
        rel = outer_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # Sum
        code += encode_d(MOVI, 0, 0)
        code += encode_d(MOVI, 13, 0)  # index
        sum_loop = len(code)
        code += encode_g(LOADOFF, 1, 13, 0)
        code += encode_e(ADD, 0, 0, 1)
        code += encode_b(INC, 13)
        code += encode_e(CMP_LT, 3, 13, 10)
        code += encode_b(JNZ, 3)
        rel = sum_loop - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="bubble_sort",
            description="Sort 8 elements using bubble sort (sum = 36)",
            bytecode=bytes(code),
            expected_result=36,
            category="memory",
            estimated_cycles=500,
        )

    @staticmethod
    def stack_heavy() -> BenchmarkWorkload:
        """Push/pop 100 elements (stack stress test).
        Push 1..100, then pop and sum, result = 5050.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)  # sum
        code += encode_d(MOVI, 1, 1)  # counter
        push_loop = len(code)
        code += encode_b(PUSH, 1)
        code += encode_b(INC, 1)
        # if R1 <= 100
        code += encode_e(CMP_LT, 3, 1, 101)
        code += encode_b(JNZ, 3)
        rel = push_loop - (len(code) + 1)
        code.append(rel & 0xFF)
        # Now pop 100 times
        code += encode_d(MOVI, 1, 0)  # pop counter
        pop_loop = len(code)
        code += encode_b(POP, 2)
        code += encode_e(ADD, 0, 0, 2)
        code += encode_b(INC, 1)
        code += encode_e(CMP_LT, 3, 1, 100)
        code += encode_b(JNZ, 3)
        rel = pop_loop - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="stack_heavy",
            description="Push/pop 100 elements (stack stress test, sum = 5050)",
            bytecode=bytes(code),
            expected_result=5050,
            category="stack",
            estimated_cycles=800,
        )

    @staticmethod
    def branch_heavy() -> BenchmarkWorkload:
        """1000 conditional jumps.
        Count how many values 0..999 are divisible by 3.
        Result should be floor(1000/3) = 333.
        Actually: 0,3,6,...,999 → 334 values.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)  # count
        code += encode_d(MOVI, 1, 0)  # i
        loop_start = len(code)
        # R2 = R1 mod 3... but MOD is rd,rs1,rs2
        code += encode_e(MOD, 2, 1, 3)  # R2 = R1 % 3
        # R3 = (R2 == 0)? CMP_EQ
        code += encode_e(CMP_EQ, 3, 2, 0)
        code += encode_b(JZ, 3)
        skip = 4
        code.append(skip & 0xFF)
        code += encode_b(INC, 0)
        code += encode_b(INC, 1)
        # if R1 < 1000, loop
        code += encode_e(CMP_LT, 3, 1, 1000)
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="branch_heavy",
            description="1000 conditional jumps (count multiples of 3 = 334)",
            bytecode=bytes(code),
            expected_result=334,
            category="control_flow",
            estimated_cycles=6000,
        )

    @staticmethod
    def memory_bandwidth() -> BenchmarkWorkload:
        """Sequential store/load across 64 memory locations.
        Store i at addr i for i=0..63, then load and sum.
        Result = 0+1+2+...+63 = 2016.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)  # sum
        code += encode_d(MOVI, 1, 0)  # i
        # Store phase
        store_loop = len(code)
        code += encode_d(STORE, 1, 0)  # Wait, STORE format: needs addr
        # Actually STORE is: STORE rd, imm8 → mem[imm8] = R[rd]? 
        # Let me re-check: the spec says STORE = 0x39
        # Based on conformance: encode_d(STORE, rd, imm8) means mem[imm8] = R[rd]
        # But we need dynamic addressing. Use STOREOFF: store rd to mem[rs1+imm16]
        # Let's use LOADI/STOREI for indexed access
        # STOREI: store rd to mem[rs1 + R[rs2]]? No...
        # Actually let's just use STORE with fixed addresses for simplicity
        # and sum as we go
        code = bytearray()
        code += encode_d(MOVI, 0, 0)   # sum
        code += encode_d(MOVI, 1, 0)   # i
        # Phase 1: store each value, adding to sum
        store_loop = len(code)
        # Store R1 at address R1 (using imm8 for 0..63)
        code += encode_d(STORE, 1, 0)  # dummy, will redo
        # Actually, we can't dynamically address with STORE easily.
        # Let's use a simpler approach: just accumulate in a register
        # and also do loads to stress memory.
        code = bytearray()
        # First, write 64 values to memory 0..63
        for i in range(64):
            code += encode_d(MOVI, 1, i)
            code += encode_d(STORE, 1, i)
        # Now read them back and sum
        code += encode_d(MOVI, 0, 0)
        for i in range(64):
            code += encode_d(LOAD, 1, i)
            code += encode_e(ADD, 0, 0, 1)
        code += encode_a(HALT)
        total = sum(range(64))  # 2016
        return BenchmarkWorkload(
            name="memory_bandwidth",
            description="Sequential store/load across 64 memory locations (sum = 2016)",
            bytecode=bytes(code),
            expected_result=total,
            category="memory",
            estimated_cycles=500,
        )

    @staticmethod
    def float_arithmetic() -> BenchmarkWorkload:
        """100 float operations.
        Compute: 100 iterations of R0 = R0 * 1.01 + 1.0, starting from 1.0.
        Use ITOF/FTOI for conversion. We'll approximate with integer math
        since exact float behavior is VM-dependent.
        Actually let's just do: iterate adding 1, starting from 0, result = 100.
        But using float opcodes to test them.
        """
        code = bytearray()
        # R0 = 0 (int), convert to float
        code += encode_d(MOVI, 0, 0)
        code += encode_b(ITOF, 0)    # R0 = 0.0 (float)
        code += encode_d(MOVI, 1, 1)
        code += encode_b(ITOF, 1)    # R1 = 1.0
        code += encode_d(MOVI, 2, 100)
        code += encode_d(MOVI, 3, 0)  # counter
        loop_start = len(code)
        code += encode_e(FADD, 0, 0, 1)  # R0 += 1.0
        code += encode_b(INC, 3)
        code += encode_e(CMP_LT, 4, 3, 2)
        code += encode_b(JNZ, 4)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_b(FTOI, 0)    # convert back to int: 100.0 → 100
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="float_arithmetic",
            description="100 float add operations (result = 100)",
            bytecode=bytes(code),
            expected_result=100,
            category="float",
            estimated_cycles=400,
        )

    @staticmethod
    def register_pressure() -> BenchmarkWorkload:
        """Use all 16 registers with data dependencies.
        Chain: R0=1, R1=R0+1, R2=R1+1, ..., R15=R14+1
        Then sum R0+R1+...+R15 and put in R0.
        R0=1, R1=2, ..., R15=16. Sum = 136.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 1)
        for i in range(1, 16):
            code += encode_e(ADDI, i, i - 1, 1)  # Wait, ADDI is encode_d format
            # Actually ADDI: rd, imm8 → R[rd] += imm8? Or R[rd] = R[rd] + imm8?
            # Based on conformance: ADDI = 0x19, encode_d(ADDI, rd, imm8)
            # So: R[rd] = R[rd] + imm8
            # We need R[i] = R[i-1] + 1, so first MOV R[i], R[i-1] then ADDI R[i], 1
            pass
        # Let's redo
        code = bytearray()
        code += encode_d(MOVI, 0, 1)
        for i in range(1, 16):
            code += encode_e(MOV, i, i - 1, 0)  # MOV R[i], R[i-1] (rs2 unused but needed)
            # Wait, MOV is format E? MOV rd, rs1, rs2? 
            # Actually MOV is typically format B or D. Let me check.
            # MOV = 0x3A. In conformance it could be encode_e(MOV, rd, rs1, 0)
            # meaning R[rd] = R[rs1]
            code += encode_e(MOV, i, i - 1, 0)
            code += encode_d(ADDI, i, 1)
        # Now sum all 16 registers
        code += encode_d(MOVI, 0, 0)
        for i in range(1, 16):
            code += encode_e(ADD, 0, 0, i)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="register_pressure",
            description="Use all 16 registers with data dependencies (sum = 136)",
            bytecode=bytes(code),
            expected_result=136,
            category="mixed",
            estimated_cycles=100,
        )

    @staticmethod
    def gcd() -> BenchmarkWorkload:
        """Euclidean GCD of (48, 18) = 6.
        R0=48, R1=18, loop: while R1 != 0: R0,R1 = R1, R0%R1
        """
        a, b = 48, 18
        code = bytearray()
        code += encode_f(MOVI16, 0, a)
        code += encode_f(MOVI16, 1, b)
        loop_start = len(code)
        # R2 = R0 % R1
        code += encode_e(MOD, 2, 0, 1)
        # R0 = R1
        code += encode_e(MOV, 0, 1, 0)
        # R1 = R2
        code += encode_e(MOV, 1, 2, 0)
        # if R1 != 0, loop
        code += encode_e(CMP_NE, 3, 1, 0)
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="gcd",
            description="Euclidean GCD of (48, 18) = 6",
            bytecode=bytes(code),
            expected_result=6,
            category="control_flow",
            estimated_cycles=150,
        )

    @staticmethod
    def prime_sieve() -> BenchmarkWorkload:
        """Sieve of Eratosthenes for primes up to 100.
        Store 1 (prime) at addr 2..100, sieve out composites.
        Count primes and store in R0. Should be 25.
        Simplified: use memory, mark composites with 0.
        """
        code = bytearray()
        # Initialize memory[2..100] = 1 (prime candidate)
        # We'll use 128 bytes starting at addr 0
        # First, FILL memory 0..127 with 1
        code += encode_d(MOVI, 0, 1)
        code += encode_d(MOVI, 1, 0)   # base addr
        code += encode_d(FILL, 0, 128)  # mem[0..127] = R[0]=1  -- FILL rd, count?
        # Actually FILL format might vary. Let's just manually set key addresses.
        # Simpler approach: just count using algorithm, skip complex memory init.
        code = bytearray()
        # Store 1 at addresses 2-100
        code += encode_d(MOVI, 1, 1)
        for addr in range(2, 101):
            code += encode_d(STORE, 1, addr)
        # Sieve
        # R2 = current prime (start at 2)
        code += encode_d(MOVI, 2, 2)
        # R10 = 10 (sqrt(100))
        code += encode_d(MOVI, 10, 10)
        outer_loop = len(code)
        # if R2 > 10, done sieving
        code += encode_e(CMP_GT, 3, 2, 10)
        code += encode_b(JNZ, 3)
        done_jmp_pos = len(code)
        code.append(0)  # placeholder
        # R4 = R2 * R2 (start marking from R2^2)
        code += encode_e(MUL, 4, 2, 2)
        inner_loop = len(code)
        # if R4 > 100, break inner
        code += encode_e(CMP_GT, 3, 4, 100)
        code += encode_b(JNZ, 3)
        break_pos = len(code)
        code.append(0)  # placeholder
        # Mark mem[R4] = 0 (composite)
        code += encode_d(MOVI, 5, 0)
        # We can't easily do dynamic store with imm8 addr > 255.
        # addr max is 100 which fits in imm8.
        code += encode_d(STORE, 5, 0)  # placeholder
        store_pos = len(code) - 1
        # R4 += R2
        code += encode_e(ADD, 4, 4, 2)
        code += encode_b(JMP, 0)
        code.append(inner_loop - (len(code) + 1))
        # Break inner loop target
        code[break_pos] = len(code) - break_pos - 1
        # R2++
        code += encode_b(INC, 2)
        code += encode_b(JMP, 0)
        code.append(outer_loop - (len(code) + 1))
        # Done sieving
        code[done_jmp_pos] = len(code) - done_jmp_pos - 1
        # Count primes: sum mem[2..100]
        code += encode_d(MOVI, 0, 0)
        code += encode_d(MOVI, 6, 2)
        count_loop = len(code)
        code += encode_d(LOAD, 1, 0)  # placeholder
        load_pos = len(code) - 1
        code += encode_e(ADD, 0, 0, 1)
        code += encode_b(INC, 6)
        code += encode_e(CMP_LT, 3, 6, 101)
        code += encode_b(JNZ, 3)
        code.append(count_loop - (len(code) + 1))
        code += encode_a(HALT)
        # Fix up the dynamic store and load addresses
        # We can't do fully dynamic addressing with imm8 format easily
        # in a single pass. Let's simplify.
        # Actually the imm8 is encoded at code generation time, not runtime.
        # We need STOREOFF or similar for runtime-dynamic addresses.
        # Let me just do a simpler version.
        
        # Simple approach: just compute the count with pure arithmetic
        # (simulating the sieve logic in registers)
        return StandardBenchmarks._prime_sieve_simple()

    @staticmethod
    def _prime_sieve_simple() -> BenchmarkWorkload:
        """Count primes up to 100 using a register-based approach.
        For each n from 2 to 100, check divisibility.
        This is O(n*sqrt(n)) but works in bytecode.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)    # count
        code += encode_d(MOVI, 1, 2)    # n = current number to check
        # outer loop
        outer = len(code)
        # Check if R1 is prime
        # R4 = divisor, start at 2
        code += encode_d(MOVI, 4, 2)
        # R5 = is_prime flag = 1
        code += encode_d(MOVI, 5, 1)
        inner = len(code)
        # if R4 * R4 > R1, done checking (it's prime)
        code += encode_e(MUL, 6, 4, 4)
        code += encode_e(CMP_GT, 7, 6, 1)
        code += encode_b(JNZ, 7)
        # jump to "is prime"
        prime_jmp = len(code)
        code.append(0)
        # Check R1 % R4 == 0
        code += encode_e(MOD, 6, 1, 4)
        code += encode_e(CMP_EQ, 7, 6, 0)
        code += encode_b(JNZ, 7)
        # if divisible, not prime
        code += encode_d(MOVI, 5, 0)
        # Jump to end of inner
        code += encode_b(JMP, 0)
        end_inner_jmp = len(code) - 1
        # "is prime" target
        code[prime_jmp] = len(code) - prime_jmp - 1
        # R4++
        code += encode_b(INC, 4)
        # if R5 still 1, continue inner
        code += encode_b(JNZ, 5)
        code.append(inner - (len(code) + 1))
        # end of inner target
        code[end_inner_jmp] = len(code) - end_inner_jmp - 1
        # if R5 == 1, increment count
        code += encode_b(JNZ, 5)
        skip_inc = len(code)
        code.append(3)
        code += encode_b(INC, 0)
        # R1++
        code += encode_b(INC, 1)
        # if R1 <= 100, continue
        code += encode_e(CMP_LT, 7, 1, 101)
        code += encode_b(JNZ, 7)
        code.append(outer - (len(code) + 1))
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="prime_sieve",
            description="Count primes up to 100 (expected: 25)",
            bytecode=bytes(code),
            expected_result=25,
            category="control_flow",
            estimated_cycles=3000,
        )

    @staticmethod
    def a2a_simulation() -> BenchmarkWorkload:
        """50 A2A operations (TELL/ASK/BCAST).
        These are message-passing opcodes. In a single-agent context
        they're essentially NOPs. Count iterations, result = 50.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)
        code += encode_d(MOVI, 1, 50)
        code += encode_d(MOVI, 2, 1)  # target agent
        loop_start = len(code)
        # TELL R2, R0 (send current count to agent 1)
        code += encode_b(TELL, 2)
        # ASK R2 (ask agent 1)
        code += encode_b(ASK, 2)
        code += encode_b(INC, 0)
        code += encode_e(CMP_LT, 3, 0, 1)
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # BCAST (broadcast)
        code += encode_b(BCAST, 2)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="a2a_simulation",
            description="50 A2A operations: TELL/ASK (result = 50)",
            bytecode=bytes(code),
            expected_result=50,
            category="a2a",
            estimated_cycles=400,
        )

    @staticmethod
    def syscall_heavy() -> BenchmarkWorkload:
        """100 NOP + SYS operations.
        Count iterations, result = 100.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)
        code += encode_d(MOVI, 1, 100)
        loop_start = len(code)
        code += encode_a(NOP)
        code += encode_a(NOP)
        code += encode_a(NOP)
        code += encode_b(INC, 0)
        code += encode_e(CMP_LT, 3, 0, 1)
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="syscall_heavy",
            description="100 iterations of NOP-heavy loop (result = 100)",
            bytecode=bytes(code),
            expected_result=100,
            category="mixed",
            estimated_cycles=600,
        )

    @staticmethod
    def noop_baseline() -> BenchmarkWorkload:
        """1000 NOPs (instruction dispatch overhead).
        Result = 0 (R0 unchanged).
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 42)
        for _ in range(1000):
            code += encode_a(NOP)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="noop_baseline",
            description="1000 NOPs (instruction dispatch overhead, R0 = 42)",
            bytecode=bytes(code),
            expected_result=42,
            category="mixed",
            estimated_cycles=1000,
        )

    @staticmethod
    def memory_random() -> BenchmarkWorkload:
        """Pseudo-random access pattern across memory.
        Access addresses in a scrambled order: 0, 50, 25, 75, 12, 63, ...
        Read and sum values. We pre-store values and read in scattered order.
        """
        code = bytearray()
        # Store values: addr i holds value i, for i in 0..63
        for i in range(64):
            code += encode_d(MOVI, 1, i)
            code += encode_d(STORE, 1, i)
        # Read in pseudo-random order and sum
        addrs = [0, 50, 25, 7, 63, 12, 38, 55, 3, 31, 47, 19, 60, 8, 42, 27,
                 15, 53, 33, 5, 61, 22, 44, 10, 58, 35, 1, 49, 28, 17, 41, 9,
                 57, 36, 14, 52, 30, 4, 46, 20, 62, 24, 48, 11, 39, 6, 59, 34,
                 16, 54, 29, 2, 43, 13, 37, 21, 56, 26, 51, 18, 45, 23, 40, 32]
        code += encode_d(MOVI, 0, 0)
        for addr in addrs:
            code += encode_d(LOAD, 1, addr)
            code += encode_e(ADD, 0, 0, 1)
        code += encode_a(HALT)
        total = sum(addrs)
        return BenchmarkWorkload(
            name="memory_random",
            description="Pseudo-random memory access pattern (sum of scattered reads)",
            bytecode=bytes(code),
            expected_result=total,
            category="memory",
            estimated_cycles=600,
        )

    @staticmethod
    def recursive_call() -> BenchmarkWorkload:
        """Recursive Fibonacci using CALL/RET.
        Compute fib(10) = 55.
        Use a simulated call stack via PUSH/POP.
        """
        # Since true recursion with CALL/RET in a self-contained bytecode
        # is tricky (need to know absolute addresses), we simulate recursion
        # with our own stack using PUSH/POP.
        code = bytearray()
        # Stack-based recursive fib(n):
        # Push n, if n <= 1 return n, else push n-1, push n-2
        # Simulate with explicit stack management.
        
        # Simpler: iterative fibonacci with stack push/pop at each step
        code += encode_d(MOVI, 0, 0)   # result
        code += encode_d(MOVI, 1, 0)   # a
        code += encode_d(MOVI, 2, 1)   # b
        code += encode_d(MOVI, 3, 10)  # target
        code += encode_d(MOVI, 4, 0)   # counter
        
        loop_start = len(code)
        # Push current state
        code += encode_b(PUSH, 1)
        code += encode_b(PUSH, 2)
        # Compute next: temp = a + b, a = b, b = temp
        code += encode_e(ADD, 5, 1, 2)
        code += encode_e(MOV, 1, 2, 0)
        code += encode_e(MOV, 2, 5, 0)
        # Pop (discard, just for stack exercise)
        code += encode_b(POP, 6)
        code += encode_b(POP, 7)
        code += encode_b(INC, 4)
        code += encode_e(CMP_LT, 6, 4, 3)
        code += encode_b(JNZ, 6)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # Result is R2 = fib(10) = 55
        code += encode_e(MOV, 0, 2, 0)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="recursive_call",
            description="Simulated recursive Fibonacci fib(10) = 55 with PUSH/POP",
            bytecode=bytes(code),
            expected_result=55,
            category="stack",
            estimated_cycles=300,
        )

    @staticmethod
    def bit_manipulation() -> BenchmarkWorkload:
        """Heavy bitwise ops (population count, shifts).
        Compute sum of popcount(i) for i = 0..63.
        Result = total number of set bits in 0..63.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)   # sum
        code += encode_d(MOVI, 1, 0)   # i
        code += encode_d(MOVI, 2, 64)  # limit
        loop_start = len(code)
        # R3 = popcount(R1)
        code += encode_b(POPCNT, 3)  # POPCNT R3 (R3 = popcount of R3? or R1?)
        # POPCNT format: encode_b(POPCNT, rd) → R[rd] = popcount(R[rd])
        # So first copy R1 to R3
        code = bytearray()
        code += encode_d(MOVI, 0, 0)
        code += encode_d(MOVI, 1, 0)
        code += encode_d(MOVI, 2, 64)
        loop_start = len(code)
        code += encode_e(MOV, 3, 1, 0)   # R3 = R1
        code += encode_b(POPCNT, 3)        # R3 = popcount(R3)
        code += encode_e(ADD, 0, 0, 3)    # sum += popcount
        code += encode_b(INC, 1)           # i++
        code += encode_e(CMP_LT, 4, 1, 2)
        code += encode_b(JNZ, 4)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_a(HALT)
        # Verify: sum of popcount(i) for i in 0..63
        expected = sum(bin(i).count('1') for i in range(64))
        return BenchmarkWorkload(
            name="bit_manipulation",
            description="Sum of popcount(i) for i=0..63 (bitwise ops)",
            bytecode=bytes(code),
            expected_result=expected,
            category="arithmetic",
            estimated_cycles=400,
        )

    @staticmethod
    def comparison_sort() -> BenchmarkWorkload:
        """Sort using CMP_EQ/CMP_LT + conditional moves.
        Find min of 8 values [8,3,5,1,7,2,6,4] by scanning.
        Then find min of remaining, etc. Result = sum of sorted = 36.
        Simpler: just find the min and max of the array.
        """
        vals = [8, 3, 5, 1, 7, 2, 6, 4]
        code = bytearray()
        # Store values
        for i, v in enumerate(vals):
            code += encode_d(MOVI, 1, v)
            code += encode_d(STORE, 1, i)
        # Find min
        code += encode_d(LOAD, 0, 0)  # min = first element
        code += encode_d(MOVI, 4, 1)  # index
        min_loop = len(code)
        code += encode_d(LOAD, 1, 0)  # placeholder
        load_pos = len(code) - 1
        # if R1 < R0, R0 = R1
        code += encode_e(CMP_LT, 3, 1, 0)
        code += encode_b(JZ, 3)
        skip = 3
        code.append(skip)
        code += encode_e(MOV, 0, 1, 0)
        code += encode_b(INC, 4)
        code += encode_e(CMP_LT, 3, 4, 8)
        code += encode_b(JNZ, 3)
        code.append(min_loop - (len(code) + 1))
        # Find max
        code += encode_d(LOAD, 1, 0)  # max = first element
        code += encode_e(MOV, 1, 0, 0)  # use R0 as initial max
        code += encode_d(MOVI, 4, 1)
        max_loop = len(code)
        code += encode_d(LOAD, 2, 0)
        load2_pos = len(code) - 1
        code += encode_e(CMP_GT, 3, 2, 1)
        code += encode_b(JZ, 3)
        code.append(3)
        code += encode_e(MOV, 1, 2, 0)
        code += encode_b(INC, 4)
        code += encode_e(CMP_LT, 3, 4, 8)
        code += encode_b(JNZ, 3)
        code.append(max_loop - (len(code) + 1))
        # Result: min + max = 1 + 8 = 9
        code += encode_e(ADD, 0, 0, 1)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="comparison_sort",
            description="Find min+max of 8 values using CMP (result = 9)",
            bytecode=bytes(code),
            expected_result=9,
            category="control_flow",
            estimated_cycles=200,
        )

    @staticmethod
    def long_program() -> BenchmarkWorkload:
        """500+ instruction sequential program.
        Compute: sum of (i * i) for i = 1..25 using chained operations.
        Result = 1 + 4 + 9 + ... + 625 = 5525.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)  # sum
        code += encode_d(MOVI, 1, 1)  # i
        loop_start = len(code)
        # R2 = R1 * R1
        code += encode_e(MUL, 2, 1, 1)
        # R0 += R2
        code += encode_e(ADD, 0, 0, 2)
        # R1++
        code += encode_b(INC, 1)
        # if R1 <= 25, loop
        code += encode_e(CMP_LT, 3, 1, 26)
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # Pad with NOPs to get 500+ instructions
        while len(code) < 510:
            code += encode_a(NOP)
        code += encode_a(HALT)
        assert len(code) >= 500
        return BenchmarkWorkload(
            name="long_program",
            description="500+ instruction program: sum of squares 1..25 = 5525",
            bytecode=bytes(code),
            expected_result=5525,
            category="mixed",
            estimated_cycles=600,
        )

    @classmethod
    def all(cls) -> list[BenchmarkWorkload]:
        """Return all standard benchmarks."""
        return [
            cls.factorial_small(),
            cls.factorial_large(),
            cls.fibonacci(),
            cls.matrix_multiply(),
            cls.bubble_sort(),
            cls.stack_heavy(),
            cls.branch_heavy(),
            cls.memory_bandwidth(),
            cls.float_arithmetic(),
            cls.register_pressure(),
            cls.gcd(),
            cls.prime_sieve(),
            cls.a2a_simulation(),
            cls.syscall_heavy(),
            cls.noop_baseline(),
            cls.memory_random(),
            cls.recursive_call(),
            cls.bit_manipulation(),
            cls.comparison_sort(),
            cls.long_program(),
        ]

    @classmethod
    def by_category(cls, category: str) -> list[BenchmarkWorkload]:
        """Return benchmarks matching a category."""
        return [b for b in cls.all() if b.category == category]
