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
        R1 = accumulator, R2 = counter. Result in R0.
        """
        code = bytearray()
        # R15 = 0 (constant zero for comparisons)
        code += encode_d(MOVI, 15, 0)
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
        # R3 = R2 != R15 → CMP_NE: R3 = (R2 != 0)
        code += encode_e(CMP_NE, 3, 2, 15)
        # JNZ R3, loop_start
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # Copy result to R0
        code += encode_e(MOV, 0, 1, 0)
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
        """Compute 12! using a loop. R1=acc, R2=counter."""
        code = bytearray()
        code += encode_d(MOVI, 15, 0)
        code += encode_d(MOVI, 1, 1)
        code += encode_d(MOVI, 2, 12)
        loop_start = len(code)
        code += encode_e(MUL, 1, 1, 2)
        code += encode_b(DEC, 2)
        code += encode_e(CMP_NE, 3, 2, 15)
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
        R0=a, R1=b, R2=counter, R3=temp, R4=target(16 for 14 iters).
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)   # a=0
        code += encode_d(MOVI, 1, 1)   # b=1
        code += encode_d(MOVI, 2, 2)   # i=2
        code += encode_d(MOVI, 4, 16)  # target: loop while R2 < 16 → 14 iterations
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
        # result in R1, copy to R0
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
        a_vals = [1,0,0, 0,1,0, 0,0,1]
        b_vals = [2,0,0, 0,3,0, 0,0,4]
        for i, v in enumerate(a_vals):
            code += encode_d(MOVI, 1, v)
            code += encode_d(STORE, 1, i)
        for i, v in enumerate(b_vals):
            code += encode_d(MOVI, 1, v)
            code += encode_d(STORE, 1, 9 + i)
        code += encode_d(MOVI, 0, 0)   # R0 = accumulator
        # C[0][0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6] = 2
        code += encode_d(LOAD, 1, 0); code += encode_d(LOAD, 2, 9)
        code += encode_e(MUL, 3, 1, 2); code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 1); code += encode_d(LOAD, 2, 12)
        code += encode_e(MUL, 3, 1, 2); code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 2); code += encode_d(LOAD, 2, 15)
        code += encode_e(MUL, 3, 1, 2); code += encode_e(ADD, 0, 0, 3)
        # C[1][1] = 3
        code += encode_d(LOAD, 1, 3); code += encode_d(LOAD, 2, 10)
        code += encode_e(MUL, 3, 1, 2); code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 4); code += encode_d(LOAD, 2, 13)
        code += encode_e(MUL, 3, 1, 2); code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 5); code += encode_d(LOAD, 2, 16)
        code += encode_e(MUL, 3, 1, 2); code += encode_e(ADD, 0, 0, 3)
        # C[2][2] = 4
        code += encode_d(LOAD, 1, 6); code += encode_d(LOAD, 2, 11)
        code += encode_e(MUL, 3, 1, 2); code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 7); code += encode_d(LOAD, 2, 14)
        code += encode_e(MUL, 3, 1, 2); code += encode_e(ADD, 0, 0, 3)
        code += encode_d(LOAD, 1, 8); code += encode_d(LOAD, 2, 17)
        code += encode_e(MUL, 3, 1, 2); code += encode_e(ADD, 0, 0, 3)
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
        return StandardBenchmarks._bubble_sort_impl()

    @staticmethod
    def _bubble_sort_impl() -> BenchmarkWorkload:
        """Actual bubble sort: sort [8,7,6,5,4,3,2,1], result = sum = 36."""
        n = 8
        vals = [8, 7, 6, 5, 4, 3, 2, 1]
        code = bytearray()
        # Store array
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
        code += encode_g(LOADOFF, 1, 12, 0)
        code += encode_g(LOADOFF, 2, 12, 1)
        code += encode_e(CMP_GT, 3, 1, 2)
        code += encode_b(JZ, 3)
        code.append(10)  # skip 10 bytes (two STOREOFF)
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
        # Sum all elements using R13 as index, R0 as accumulator
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
        """Push/pop 50 elements (stack stress test).
        Push 1..50, then pop and sum, result = 1275.
        Using 50 to stay within the 64-deep stack limit.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)   # sum
        code += encode_d(MOVI, 1, 1)   # counter
        code += encode_d(MOVI, 10, 51)  # push limit (compare R1 < R10)
        push_loop = len(code)
        code += encode_b(PUSH, 1)
        code += encode_b(INC, 1)
        code += encode_e(CMP_LT, 3, 1, 10)  # R3 = (R1 < 51)
        code += encode_b(JNZ, 3)
        rel = push_loop - (len(code) + 1)
        code.append(rel & 0xFF)
        # Now pop 50 times
        code += encode_d(MOVI, 1, 0)   # pop counter
        code += encode_d(MOVI, 10, 50)  # pop limit
        pop_loop = len(code)
        code += encode_b(POP, 2)
        code += encode_e(ADD, 0, 0, 2)
        code += encode_b(INC, 1)
        code += encode_e(CMP_LT, 3, 1, 10)  # R3 = (R1 < 50)
        code += encode_b(JNZ, 3)
        rel = pop_loop - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="stack_heavy",
            description="Push/pop 50 elements (stack stress test, sum = 1275)",
            bytecode=bytes(code),
            expected_result=1275,
            category="stack",
            estimated_cycles=400,
        )

    @staticmethod
    def branch_heavy() -> BenchmarkWorkload:
        """1000 conditional jumps.
        Count how many values 0..999 are divisible by 3.
        Result: 334 values (0,3,6,...,999).
        Uses R15 as constant zero for comparisons.
        """
        code = bytearray()
        code += encode_d(MOVI, 15, 0)   # constant zero
        code += encode_d(MOVI, 0, 0)   # count
        code += encode_d(MOVI, 1, 0)   # i
        code += encode_d(MOVI, 11, 3)   # divisor constant
        code += encode_f(MOVI16, 10, 1000)  # loop limit
        loop_start = len(code)
        # R2 = R1 % R11 (divisor=3)
        code += encode_e(MOD, 2, 1, 11)
        # R3 = (R2 == 0)?
        code += encode_e(CMP_EQ, 3, 2, 15)  # compare with R15(=0)
        # Always increment i
        code += encode_b(INC, 1)
        # If R3 == 0 (not divisible), skip INC R0
        code += encode_b(JZ, 3)
        code.append(2)  # skip INC R0 (2 bytes)
        code += encode_b(INC, 0)
        # if R1 < 1000, loop (CMP_LT R3, R1, R10)
        code += encode_e(CMP_LT, 3, 1, 10)
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
        Iteratively add 1.0 (as float) 100 times, starting from 0.0.
        Result = 100.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)
        code += encode_b(ITOF, 0)    # R0 = 0.0 (float)
        code += encode_d(MOVI, 1, 1)
        code += encode_b(ITOF, 1)    # R1 = 1.0
        code += encode_d(MOVI, 2, 100)  # limit
        code += encode_d(MOVI, 3, 0)  # counter
        loop_start = len(code)
        code += encode_e(FADD, 0, 0, 1)  # R0 += 1.0
        code += encode_b(INC, 3)
        code += encode_e(CMP_LT, 4, 3, 2)  # counter < limit
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
        Chain: R1=1, R2=R1+1, R3=R2+1, ..., R15=R14+1
        Then sum R1+R2+...+R15 and put in R0.
        R1=1, R2=2, ..., R15=15. Sum = 1+2+...+15 = 120.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)   # sum accumulator
        code += encode_d(MOVI, 1, 1)   # first value
        # Build chain: R[i] = R[i-1] + 1 for i = 2..15
        for i in range(2, 16):
            code += encode_e(MOV, i, i - 1, 0)  # R[i] = R[i-1]
            code += encode_d(ADDI, i, 1)           # R[i] += 1
        # Sum R1..R15 into R0
        for i in range(1, 16):
            code += encode_e(ADD, 0, 0, i)
        code += encode_a(HALT)
        return BenchmarkWorkload(
            name="register_pressure",
            description="Use all 16 registers with data dependencies (sum = 120)",
            bytecode=bytes(code),
            expected_result=120,
            category="mixed",
            estimated_cycles=100,
        )

    @staticmethod
    def gcd() -> BenchmarkWorkload:
        """Euclidean GCD of (48, 18) = 6.
        R0=a, R1=b, loop: while R1 != 0: R0,R1 = R1, R0%R1
        Uses R15=0 for comparison.
        """
        a, b = 48, 18
        code = bytearray()
        code += encode_f(MOVI16, 0, a)
        code += encode_f(MOVI16, 1, b)
        code += encode_d(MOVI, 15, 0)  # constant zero
        loop_start = len(code)
        # R2 = R0 % R1
        code += encode_e(MOD, 2, 0, 1)
        # R0 = R1
        code += encode_e(MOV, 0, 1, 0)
        # R1 = R2
        code += encode_e(MOV, 1, 2, 0)
        # if R1 != R15(=0), loop
        code += encode_e(CMP_NE, 3, 1, 15)
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
        """Count primes up to 100 using trial division.
        For each n from 2 to 100, check divisibility by 2..sqrt(n).
        Uses R15=0 for zero comparisons.
        """
        return StandardBenchmarks._prime_sieve_impl()

    @staticmethod
    def _prime_sieve_impl() -> BenchmarkWorkload:
        """Count primes up to 100 using trial division."""
        code = bytearray()
        code += encode_d(MOVI, 15, 0)   # constant zero
        code += encode_d(MOVI, 0, 0)    # count
        code += encode_d(MOVI, 1, 2)    # n
        # outer loop
        outer = len(code)
        # R4 = divisor, start at 2
        code += encode_d(MOVI, 4, 2)
        # R5 = is_prime flag = 1
        code += encode_d(MOVI, 5, 1)
        inner = len(code)
        # if R4 * R4 > R1, done checking (it's prime) → goto end_inner
        code += encode_e(MUL, 6, 4, 4)
        code += encode_e(CMP_GT, 7, 6, 1)
        code += encode_b(JNZ, 7)
        end_inner_jmp1 = len(code)
        code.append(0)  # placeholder: jump to end_inner
        # Check R1 % R4 == 0
        code += encode_e(MOD, 6, 1, 4)
        code += encode_e(CMP_EQ, 7, 6, 15)  # compare with R15(=0)
        code += encode_b(JZ, 7)
        # If divisible (R7=1): JZ not taken → fall through to not_prime section
        # If not divisible (R7=0): JZ taken → skip not_prime section → goto next_divisor
        # not_prime section = MOVI R5,0 (3B) + JMP (3B) = 6 bytes to skip
        code.append(6)
        # not_prime: mark as not prime and jump to end_inner
        code += encode_d(MOVI, 5, 0)
        code += encode_b(JMP, 0)
        end_inner_jmp2 = len(code)
        code.append(0)  # placeholder: jump to end_inner
        # next_divisor:
        code += encode_b(INC, 4)       # R4++
        code += encode_b(JNZ, 5)       # R5==1 → continue inner loop
        code.append((inner - (len(code) + 1)) & 0xFF)
        # end_inner:
        code[end_inner_jmp1] = (len(code) - end_inner_jmp1 - 1) & 0xFF
        code[end_inner_jmp2] = (len(code) - end_inner_jmp2 - 1) & 0xFF
        # if R5 != 0 (is prime), increment count
        code += encode_b(JZ, 5)        # if R5==0, skip INC R0
        code.append(2)                  # skip 2 bytes (INC R0)
        code += encode_b(INC, 0)
        # R1++
        code += encode_b(INC, 1)
        # if R1 < 101, continue outer
        code += encode_d(MOVI, 10, 101)  # limit
        code += encode_e(CMP_LT, 7, 1, 10)
        code += encode_b(JNZ, 7)
        code.append((outer - (len(code) + 1)) & 0xFF)
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
        Count iterations, result = 50.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)
        code += encode_d(MOVI, 1, 50)  # limit
        code += encode_d(MOVI, 2, 1)   # target agent
        loop_start = len(code)
        code += encode_b(TELL, 2)
        code += encode_b(ASK, 2)
        code += encode_b(INC, 0)
        code += encode_e(CMP_LT, 3, 0, 1)  # R0 < R1(=50)
        code += encode_b(JNZ, 3)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
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
        code += encode_d(MOVI, 1, 100)  # limit
        loop_start = len(code)
        code += encode_a(NOP)
        code += encode_a(NOP)
        code += encode_a(NOP)
        code += encode_b(INC, 0)
        code += encode_e(CMP_LT, 3, 0, 1)  # R0 < R1(=100)
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
        Result = 42 (R0 unchanged).
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
        Read values in scrambled order and sum them.
        """
        code = bytearray()
        for i in range(64):
            code += encode_d(MOVI, 1, i)
            code += encode_d(STORE, 1, i)
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
        """Simulated recursive Fibonacci fib(10) = 55 with PUSH/POP.
        Iterative fibonacci with stack push/pop at each step.
        After 10 iterations: R1 = fib(10) = 55.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)   # unused
        code += encode_d(MOVI, 1, 0)   # a
        code += encode_d(MOVI, 2, 1)   # b
        code += encode_d(MOVI, 3, 10)  # loop limit
        code += encode_d(MOVI, 4, 0)   # counter
        loop_start = len(code)
        code += encode_b(PUSH, 1)
        code += encode_b(PUSH, 2)
        code += encode_e(ADD, 5, 1, 2)  # temp = a + b
        code += encode_e(MOV, 1, 2, 0)   # a = b
        code += encode_e(MOV, 2, 5, 0)   # b = temp
        code += encode_b(POP, 6)
        code += encode_b(POP, 7)
        code += encode_b(INC, 4)
        code += encode_e(CMP_LT, 6, 4, 3)  # counter < limit
        code += encode_b(JNZ, 6)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        # Result: R1 = fib(10) = 55
        code += encode_e(MOV, 0, 1, 0)
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
        """Heavy bitwise ops (population count).
        Compute sum of popcount(i) for i = 0..63.
        """
        code = bytearray()
        code += encode_d(MOVI, 0, 0)   # sum
        code += encode_d(MOVI, 1, 0)   # i
        code += encode_d(MOVI, 2, 64)  # limit
        loop_start = len(code)
        code += encode_e(MOV, 3, 1, 0)   # R3 = R1 (copy)
        code += encode_b(POPCNT, 3)        # R3 = popcount(R3)
        code += encode_e(ADD, 0, 0, 3)    # sum += popcount
        code += encode_b(INC, 1)           # i++
        code += encode_e(CMP_LT, 4, 1, 2)  # i < limit
        code += encode_b(JNZ, 4)
        rel = loop_start - (len(code) + 1)
        code.append(rel & 0xFF)
        code += encode_a(HALT)
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
        """Find min+max of 8 values [8,3,5,1,7,2,6,4] using CMP ops.
        Result = min + max = 1 + 8 = 9.
        """
        vals = [8, 3, 5, 1, 7, 2, 6, 4]
        code = bytearray()
        # Store values at addresses 0-7
        for i, v in enumerate(vals):
            code += encode_d(MOVI, 1, v)
            code += encode_d(STORE, 1, i)
        # Find min: scan memory[1..7], track minimum in R0
        code += encode_d(MOVI, 4, 0)         # R4 = index = 0
        code += encode_g(LOADOFF, 0, 4, 0)   # R0 = mem[0] = 8 (initial min)
        code += encode_b(INC, 4)              # R4 = 1
        code += encode_d(MOVI, 8, 8)          # R8 = limit = 8
        min_loop = len(code)
        code += encode_g(LOADOFF, 1, 4, 0)   # R1 = mem[R4]
        code += encode_e(CMP_LT, 3, 1, 0)    # R3 = (R1 < R0)
        code += encode_b(JZ, 3)              # skip if not less
        code.append(4)                        # skip MOV (4 bytes)
        code += encode_e(MOV, 0, 1, 0)       # R0 = R1 (update min)
        code += encode_b(INC, 4)
        code += encode_e(CMP_LT, 3, 4, 8)    # R4 < R8
        code += encode_b(JNZ, 3)
        rel = min_loop - (len(code) + 1)
        code.append(rel & 0xFF)
        # Find max: scan memory[1..7], track maximum in R1
        code += encode_d(MOVI, 4, 0)         # R4 = index = 0
        code += encode_g(LOADOFF, 1, 4, 0)   # R1 = mem[0] = 8 (initial max)
        code += encode_b(INC, 4)              # R4 = 1
        max_loop = len(code)
        code += encode_g(LOADOFF, 2, 4, 0)   # R2 = mem[R4]
        code += encode_e(CMP_GT, 3, 2, 1)    # R3 = (R2 > R1)
        code += encode_b(JZ, 3)              # skip if not greater
        code.append(4)                        # skip MOV (4 bytes)
        code += encode_e(MOV, 1, 2, 0)       # R1 = R2 (update max)
        code += encode_b(INC, 4)
        code += encode_e(CMP_LT, 3, 4, 8)    # R4 < R8
        code += encode_b(JNZ, 3)
        rel = max_loop - (len(code) + 1)
        code.append(rel & 0xFF)
        # Result: R0 = min, R1 = max, R0 = min + max
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
        code += encode_d(MOVI, 0, 0)   # sum
        code += encode_d(MOVI, 1, 1)   # i
        code += encode_d(MOVI, 10, 26)  # limit: loop while R1 < 26
        loop_start = len(code)
        code += encode_e(MUL, 2, 1, 1)   # R2 = R1 * R1
        code += encode_e(ADD, 0, 0, 2)   # R0 += R2
        code += encode_b(INC, 1)          # R1++
        code += encode_e(CMP_LT, 3, 1, 10)  # R1 < R10(=26)
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
