"""VM Adapter interface for the FLUX profiler.

Provides an abstract protocol for VM implementations and a concrete
MiniVMAdapter that executes ISA v2 bytecode directly.
"""

from __future__ import annotations

import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# ISA v2 opcodes (subset needed for execution)
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

# Cycle costs for different instruction categories
CYCLE_SIMPLE = 1
CYCLE_ARITH = 2
CYCLE_MEMORY = 3
CYCLE_FLOAT = 4
CYCLE_BRANCH = 2
CYCLE_A2A = 5

_CYCLE_MAP: dict[int, int] = {
    NOP: CYCLE_SIMPLE,
    HALT: CYCLE_SIMPLE,
    RET: CYCLE_SIMPLE,
    INC: CYCLE_SIMPLE,
    DEC: CYCLE_SIMPLE,
    NOT: CYCLE_SIMPLE,
    NEG: CYCLE_SIMPLE,
    MOVI: CYCLE_SIMPLE,
    MOV: CYCLE_SIMPLE,
    ADDI: CYCLE_ARITH,
    SUBI: CYCLE_ARITH,
    ANDI: CYCLE_ARITH,
    ORI: CYCLE_ARITH,
    XORI: CYCLE_ARITH,
    SHLI: CYCLE_ARITH,
    SHRI: CYCLE_ARITH,
    ADDI16: CYCLE_ARITH,
    SUBI16: CYCLE_ARITH,
    MOVI16: CYCLE_SIMPLE,
    ADD: CYCLE_ARITH,
    SUB: CYCLE_ARITH,
    MUL: CYCLE_ARITH,
    DIV: CYCLE_ARITH,
    MOD: CYCLE_ARITH,
    AND: CYCLE_ARITH,
    OR: CYCLE_ARITH,
    XOR: CYCLE_ARITH,
    SHL: CYCLE_ARITH,
    SHR: CYCLE_ARITH,
    MIN: CYCLE_ARITH,
    MAX: CYCLE_ARITH,
    CMP_EQ: CYCLE_ARITH,
    CMP_LT: CYCLE_ARITH,
    CMP_GT: CYCLE_ARITH,
    CMP_NE: CYCLE_ARITH,
    FADD: CYCLE_FLOAT,
    FSUB: CYCLE_FLOAT,
    FMUL: CYCLE_FLOAT,
    FDIV: CYCLE_FLOAT,
    FTOI: CYCLE_FLOAT,
    ITOF: CYCLE_FLOAT,
    LOAD: CYCLE_MEMORY,
    STORE: CYCLE_MEMORY,
    LOADOFF: CYCLE_MEMORY,
    STOREOFF: CYCLE_MEMORY,
    LOADI: CYCLE_MEMORY,
    STOREI: CYCLE_MEMORY,
    PUSH: CYCLE_SIMPLE,
    POP: CYCLE_SIMPLE,
    JZ: CYCLE_BRANCH,
    JNZ: CYCLE_BRANCH,
    JLT: CYCLE_BRANCH,
    JGT: CYCLE_BRANCH,
    JMP: CYCLE_BRANCH,
    JAL: CYCLE_BRANCH,
    CALL: CYCLE_BRANCH,
    LOOP: CYCLE_BRANCH,
    SWP: CYCLE_SIMPLE,
    ENTER: CYCLE_SIMPLE,
    LEAVE: CYCLE_SIMPLE,
    COPY: CYCLE_MEMORY,
    FILL: CYCLE_MEMORY,
    TELL: CYCLE_A2A,
    ASK: CYCLE_A2A,
    DELEG: CYCLE_A2A,
    BCAST: CYCLE_A2A,
    ABS: CYCLE_ARITH,
    CLZ: CYCLE_ARITH,
    CTZ: CYCLE_ARITH,
    POPCNT: CYCLE_ARITH,
}


def get_cycle_cost(opcode: int) -> int:
    """Get the estimated cycle cost for an opcode."""
    return _CYCLE_MAP.get(opcode, CYCLE_SIMPLE)


@dataclass
class ProfileResult:
    """Result of profiling a single benchmark execution."""
    benchmark_name: str
    vm_name: str
    wall_time_ns: int
    instructions_executed: int
    cycles_used: int
    memory_reads: int
    memory_writes: int
    peak_stack_depth: int
    registers_final: dict[int, int]
    result_correct: bool
    error: Optional[str] = None


class VMAdapter(ABC):
    """Abstract interface for VM implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this VM implementation."""
        ...

    @abstractmethod
    def execute(self, bytecode: bytes, timeout: float = 10.0) -> ProfileResult:
        """Execute bytecode and return profiling results."""
        ...


class MiniVM:
    """Minimal ISA v2 virtual machine for benchmark execution.

    Features:
    - 16 general-purpose 32-bit registers
    - 256-byte memory (byte-addressable)
    - 64-deep call/value stack
    - Cycle counting, instruction counting
    - Memory access counting
    - Peak stack depth tracking
    """

    MAX_REGISTERS = 16
    MEMORY_SIZE = 256
    STACK_SIZE = 64
    MAX_INSTRUCTIONS = 10_000_000  # safety limit

    def __init__(self) -> None:
        self.registers: list[int] = [0] * self.MAX_REGISTERS
        self.memory: bytearray = bytearray(self.MEMORY_SIZE)
        self.stack: list[int] = [0] * self.STACK_SIZE
        self.sp: int = 0  # stack pointer (grows up)
        self.pc: int = 0  # program counter
        self.halted: bool = False
        self.error: Optional[str] = None
        self.instructions_executed: int = 0
        self.cycles_used: int = 0
        self.memory_reads: int = 0
        self.memory_writes: int = 0
        self.peak_stack_depth: int = 0
        self.bytecode: bytes = b""

    def _sign_extend_8(self, val: int) -> int:
        """Sign extend an 8-bit value to Python int."""
        val = val & 0xFF
        if val >= 128:
            return val - 256
        return val

    def _to_i32(self, val: int) -> int:
        """Clamp value to signed 32-bit range."""
        val = val & 0xFFFFFFFF
        if val >= 0x80000000:
            return val - 0x100000000
        return val

    def _to_u32(self, val: int) -> int:
        """Clamp value to unsigned 32-bit range."""
        return val & 0xFFFFFFFF

    def _to_u8(self, val: int) -> int:
        """Clamp value to unsigned 8-bit range."""
        return val & 0xFF

    def _push(self, val: int) -> None:
        """Push value onto the stack."""
        if self.sp >= self.STACK_SIZE:
            self.error = "Stack overflow"
            self.halted = True
            return
        self.stack[self.sp] = self._to_i32(val)
        self.sp += 1
        if self.sp > self.peak_stack_depth:
            self.peak_stack_depth = self.sp

    def _pop(self) -> int:
        """Pop value from the stack."""
        if self.sp <= 0:
            self.error = "Stack underflow"
            self.halted = True
            return 0
        self.sp -= 1
        return self.stack[self.sp]

    def _read8(self, pc: int) -> int:
        """Read 8-bit value from bytecode at pc."""
        if pc < len(self.bytecode):
            return self.bytecode[pc]
        self.error = f"PC out of bounds: {pc}"
        self.halted = True
        return 0

    def _read16_le(self, pc: int) -> int:
        """Read 16-bit little-endian value from bytecode at pc."""
        lo = self._read8(pc)
        hi = self._read8(pc + 1)
        return lo | (hi << 8)

    def _mem_read(self, addr: int) -> int:
        """Read a byte from memory."""
        addr = self._to_u8(addr)
        self.memory_reads += 1
        return self.memory[addr]

    def _mem_write(self, addr: int, val: int) -> None:
        """Write a byte to memory."""
        addr = self._to_u8(addr)
        self.memory_writes += 1
        self.memory[addr] = self._to_u8(val)

    def run(self, bytecode: bytes, timeout: float = 10.0) -> None:
        """Execute the given bytecode."""
        self.bytecode = bytecode
        self.pc = 0
        self.halted = False
        self.error = None
        start_time = time.perf_counter()

        while not self.halted and self.pc < len(bytecode):
            # Check timeout
            if (time.perf_counter() - start_time) > timeout:
                self.error = "Timeout"
                self.halted = True
                break

            # Safety check
            if self.instructions_executed >= self.MAX_INSTRUCTIONS:
                self.error = "Instruction limit exceeded"
                self.halted = True
                break

            self._step()

        if self.error is None and self.pc >= len(bytecode):
            self.error = "Program fell off end without HALT"
            self.halted = True

    def _step(self) -> None:
        """Execute a single instruction."""
        opcode = self._read8(self.pc)
        self.instructions_executed += 1
        self.cycles_used += get_cycle_cost(opcode)

        # Determine format and dispatch
        if opcode == HALT:
            self.halted = True
            return

        if opcode == NOP:
            self.pc += 1
            return

        # Format A: single byte (0x00-0x07, but some are used)
        if opcode in (RET,):
            # Return: pop PC from stack
            if self.sp > 0:
                self.sp -= 1
                self.pc = self.stack[self.sp]
                return
            else:
                self.error = "Stack underflow on RET"
                self.halted = True
                return

        # Format B: opcode + rd (1 byte)
        if opcode in (INC, DEC, NOT, NEG, PUSH, POP, JZ, JNZ, JLT, JGT, JMP,
                      JAL, FTOI, ITOF, ABS, CLZ, CTZ, POPCNT,
                      TELL, ASK, DELEG, BCAST):
            if self.pc + 1 >= len(self.bytecode):
                self.error = "Truncated format B instruction"
                self.halted = True
                return
            rd = self._read8(self.pc + 1) & 0xF
            self._exec_b(opcode, rd)
            return

        # Format C: opcode + imm8
        if opcode in (0x03, 0x04, 0x05, 0x06, 0x07,
                      0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17):
            self.pc += 2
            return  # unimplemented, skip

        # Format D: opcode + rd + imm8
        if opcode in (MOVI, ADDI, SUBI, ANDI, ORI, XORI, SHLI, SHRI,
                      LOAD, STORE):
            if self.pc + 2 >= len(self.bytecode):
                self.error = f"Truncated format D instruction at {self.pc}"
                self.halted = True
                return
            rd = self._read8(self.pc + 1) & 0xF
            imm8 = self._read8(self.pc + 2)
            self._exec_d(opcode, rd, imm8)
            return

        # Format E: opcode + rd + rs1 + rs2
        if opcode in (ADD, SUB, MUL, DIV, MOD, AND, OR, XOR, SHL, SHR,
                      MIN, MAX, CMP_EQ, CMP_LT, CMP_GT, CMP_NE,
                      FADD, FSUB, FMUL, FDIV,
                      MOV, SWP):
            if self.pc + 3 >= len(self.bytecode):
                self.error = f"Truncated format E instruction at {self.pc}"
                self.halted = True
                return
            rd = self._read8(self.pc + 1) & 0xF
            rs1 = self._read8(self.pc + 2) & 0xF
            rs2 = self._read8(self.pc + 3) & 0xF
            self._exec_e(opcode, rd, rs1, rs2)
            return

        # Format F: opcode + rd + imm16 (LE)
        if opcode in (MOVI16, ADDI16, SUBI16):
            if self.pc + 3 >= len(self.bytecode):
                self.error = f"Truncated format F instruction at {self.pc}"
                self.halted = True
                return
            rd = self._read8(self.pc + 1) & 0xF
            imm16 = self._read16_le(self.pc + 2)
            self._exec_f(opcode, rd, imm16)
            return

        # Format G: opcode + rd + rs1 + imm16 (LE)
        if opcode in (LOADOFF, STOREOFF):
            if self.pc + 4 >= len(self.bytecode):
                self.error = f"Truncated format G instruction at {self.pc}"
                self.halted = True
                return
            rd = self._read8(self.pc + 1) & 0xF
            rs1 = self._read8(self.pc + 2) & 0xF
            imm16 = self._read16_le(self.pc + 3)
            self._exec_g(opcode, rd, rs1, imm16)
            return

        # Other formats
        if opcode in (CALL, LOOP, ENTER, LEAVE, COPY, FILL, LOADI, STOREI):
            if self.pc + 1 >= len(self.bytecode):
                self.error = f"Truncated instruction at {self.pc}"
                self.halted = True
                return
            rd = self._read8(self.pc + 1) & 0xF
            self._exec_misc(opcode, rd)
            return

        # Unknown opcode: skip
        self.pc += 1

    def _exec_b(self, opcode: int, rd: int) -> None:
        """Execute format B instructions: opcode + rd."""
        if opcode == INC:
            self.registers[rd] = self._to_i32(self.registers[rd] + 1)
            self.pc += 2
        elif opcode == DEC:
            self.registers[rd] = self._to_i32(self.registers[rd] - 1)
            self.pc += 2
        elif opcode == NOT:
            self.registers[rd] = self._to_i32(~self.registers[rd])
            self.pc += 2
        elif opcode == NEG:
            self.registers[rd] = self._to_i32(-self.registers[rd])
            self.pc += 2
        elif opcode == PUSH:
            self._push(self.registers[rd])
            self.pc += 2
        elif opcode == POP:
            self.registers[rd] = self._pop()
            self.pc += 2
        elif opcode == JZ:
            if self.pc + 2 >= len(self.bytecode):
                self.error = "Truncated JZ"
                self.halted = True
                return
            offset = self._sign_extend_8(self._read8(self.pc + 2))
            self.pc += 3
            if self.registers[rd] == 0:
                self.pc = self._to_u32(self.pc + offset)
        elif opcode == JNZ:
            if self.pc + 2 >= len(self.bytecode):
                self.error = "Truncated JNZ"
                self.halted = True
                return
            offset = self._sign_extend_8(self._read8(self.pc + 2))
            self.pc += 3
            if self.registers[rd] != 0:
                self.pc = self._to_u32(self.pc + offset)
        elif opcode == JLT:
            if self.pc + 2 >= len(self.bytecode):
                self.error = "Truncated JLT"
                self.halted = True
                return
            offset = self._sign_extend_8(self._read8(self.pc + 2))
            self.pc += 3
            if self.registers[rd] < 0:
                self.pc = self._to_u32(self.pc + offset)
        elif opcode == JGT:
            if self.pc + 2 >= len(self.bytecode):
                self.error = "Truncated JGT"
                self.halted = True
                return
            offset = self._sign_extend_8(self._read8(self.pc + 2))
            self.pc += 3
            if self.registers[rd] > 0:
                self.pc = self._to_u32(self.pc + offset)
        elif opcode == JMP:
            if self.pc + 2 >= len(self.bytecode):
                self.error = "Truncated JMP"
                self.halted = True
                return
            offset = self._sign_extend_8(self._read8(self.pc + 2))
            self.pc = self._to_u32(self.pc + 3 + offset)
        elif opcode == JAL:
            if self.pc + 2 >= len(self.bytecode):
                self.error = "Truncated JAL"
                self.halted = True
                return
            offset = self._sign_extend_8(self._read8(self.pc + 2))
            self._push(self.pc + 3)
            self.pc = self._to_u32(self.pc + 3 + offset)
        elif opcode == FTOI:
            # Float-to-int: interpret register bits as float, convert
            bits = self._to_u32(self.registers[rd])
            try:
                fval = struct.unpack('!f', struct.pack('!I', bits))[0]
                self.registers[rd] = self._to_i32(int(fval))
            except (struct.error, OverflowError):
                self.registers[rd] = 0
            self.pc += 2
        elif opcode == ITOF:
            # Int-to-float: interpret register as int, convert to float bits
            ival = self.registers[rd]
            try:
                bits = struct.unpack('!I', struct.pack('!f', float(ival)))[0]
                self.registers[rd] = self._to_i32(bits)
            except (struct.error, OverflowError):
                self.registers[rd] = 0
            self.pc += 2
        elif opcode == ABS:
            val = self.registers[rd]
            if val < 0:
                self.registers[rd] = -val
            self.pc += 2
        elif opcode == CLZ:
            val = self._to_u32(self.registers[rd])
            if val == 0:
                self.registers[rd] = 32
            else:
                self.registers[rd] = 32 - val.bit_length()
            self.pc += 2
        elif opcode == CTZ:
            val = self._to_u32(self.registers[rd])
            if val == 0:
                self.registers[rd] = 32
            else:
                self.registers[rd] = (val & -val).bit_length() - 1
            self.pc += 2
        elif opcode == POPCNT:
            val = self._to_u32(self.registers[rd])
            self.registers[rd] = bin(val).count('1')
            self.pc += 2
        elif opcode in (TELL, ASK, DELEG, BCAST):
            # A2A opcodes: no-op in single-agent context
            self.pc += 2
        else:
            self.pc += 2

    def _exec_d(self, opcode: int, rd: int, imm8: int) -> None:
        """Execute format D instructions: opcode + rd + imm8."""
        simm = self._sign_extend_8(imm8)
        if opcode == MOVI:
            self.registers[rd] = simm
            self.pc += 3
        elif opcode == ADDI:
            self.registers[rd] = self._to_i32(self.registers[rd] + simm)
            self.pc += 3
        elif opcode == SUBI:
            self.registers[rd] = self._to_i32(self.registers[rd] - simm)
            self.pc += 3
        elif opcode == ANDI:
            self.registers[rd] = self._to_i32(self._to_u32(self.registers[rd]) & self._to_u32(simm))
            self.pc += 3
        elif opcode == ORI:
            self.registers[rd] = self._to_i32(self._to_u32(self.registers[rd]) | self._to_u32(simm))
            self.pc += 3
        elif opcode == XORI:
            self.registers[rd] = self._to_i32(self._to_u32(self.registers[rd]) ^ self._to_u32(simm))
            self.pc += 3
        elif opcode == SHLI:
            self.registers[rd] = self._to_i32(self._to_u32(self.registers[rd]) << (simm & 0x1F))
            self.pc += 3
        elif opcode == SHRI:
            self.registers[rd] = self._to_i32(self._to_u32(self.registers[rd]) >> (simm & 0x1F))
            self.pc += 3
        elif opcode == LOAD:
            addr = imm8
            self.registers[rd] = self._sign_extend_8(self._mem_read(addr))
            self.pc += 3
        elif opcode == STORE:
            addr = imm8
            self._mem_write(addr, self.registers[rd])
            self.pc += 3
        else:
            self.pc += 3

    def _exec_e(self, opcode: int, rd: int, rs1: int, rs2: int) -> None:
        """Execute format E instructions: opcode + rd + rs1 + rs2."""
        a = self.registers[rs1]
        b = self.registers[rs2]
        ua = self._to_u32(a)
        ub = self._to_u32(b)

        if opcode == MOV:
            self.registers[rd] = a
            self.pc += 4
        elif opcode == SWP:
            self.registers[rd] = b
            self.registers[rs1] = a
            self.pc += 4
        elif opcode == ADD:
            self.registers[rd] = self._to_i32(a + b)
            self.pc += 4
        elif opcode == SUB:
            self.registers[rd] = self._to_i32(a - b)
            self.pc += 4
        elif opcode == MUL:
            self.registers[rd] = self._to_i32(a * b)
            self.pc += 4
        elif opcode == DIV:
            if b == 0:
                self.error = "Division by zero"
                self.halted = True
                return
            # Python integer division rounds toward negative infinity;
            # we want truncation toward zero
            result = int(a / b) if (a < 0) != (b < 0) else a // b
            self.registers[rd] = self._to_i32(result)
            self.pc += 4
        elif opcode == MOD:
            if b == 0:
                self.error = "Modulo by zero"
                self.halted = True
                return
            # Match C semantics: result has sign of dividend
            result = a - int(a / b) * b if (a < 0) != (b < 0) else a % b
            self.registers[rd] = self._to_i32(result)
            self.pc += 4
        elif opcode == AND:
            self.registers[rd] = self._to_i32(ua & ub)
            self.pc += 4
        elif opcode == OR:
            self.registers[rd] = self._to_i32(ua | ub)
            self.pc += 4
        elif opcode == XOR:
            self.registers[rd] = self._to_i32(ua ^ ub)
            self.pc += 4
        elif opcode == SHL:
            self.registers[rd] = self._to_i32(ua << (ub & 0x1F))
            self.pc += 4
        elif opcode == SHR:
            self.registers[rd] = self._to_i32(ua >> (ub & 0x1F))
            self.pc += 4
        elif opcode == MIN:
            self.registers[rd] = a if a < b else b
            self.pc += 4
        elif opcode == MAX:
            self.registers[rd] = a if a > b else b
            self.pc += 4
        elif opcode == CMP_EQ:
            self.registers[rd] = 1 if a == b else 0
            self.pc += 4
        elif opcode == CMP_LT:
            self.registers[rd] = 1 if a < b else 0
            self.pc += 4
        elif opcode == CMP_GT:
            self.registers[rd] = 1 if a > b else 0
            self.pc += 4
        elif opcode == CMP_NE:
            self.registers[rd] = 1 if a != b else 0
            self.pc += 4
        elif opcode == FADD:
            # Interpret a and b as IEEE 754 floats
            try:
                fa = struct.unpack('!f', struct.pack('!I', ua))[0]
                fb = struct.unpack('!f', struct.pack('!I', ub))[0]
                result = struct.unpack('!I', struct.pack('!f', fa + fb))[0]
                self.registers[rd] = self._to_i32(result)
            except (struct.error, OverflowError):
                self.registers[rd] = 0
            self.pc += 4
        elif opcode == FSUB:
            try:
                fa = struct.unpack('!f', struct.pack('!I', ua))[0]
                fb = struct.unpack('!f', struct.pack('!I', ub))[0]
                result = struct.unpack('!I', struct.pack('!f', fa - fb))[0]
                self.registers[rd] = self._to_i32(result)
            except (struct.error, OverflowError):
                self.registers[rd] = 0
            self.pc += 4
        elif opcode == FMUL:
            try:
                fa = struct.unpack('!f', struct.pack('!I', ua))[0]
                fb = struct.unpack('!f', struct.pack('!I', ub))[0]
                result = struct.unpack('!I', struct.pack('!f', fa * fb))[0]
                self.registers[rd] = self._to_i32(result)
            except (struct.error, OverflowError):
                self.registers[rd] = 0
            self.pc += 4
        elif opcode == FDIV:
            try:
                fa = struct.unpack('!f', struct.pack('!I', ua))[0]
                fb = struct.unpack('!f', struct.pack('!I', ub))[0]
                if fb == 0.0:
                    self.error = "Float division by zero"
                    self.halted = True
                    return
                result = struct.unpack('!I', struct.pack('!f', fa / fb))[0]
                self.registers[rd] = self._to_i32(result)
            except (struct.error, OverflowError):
                self.registers[rd] = 0
            self.pc += 4
        else:
            self.pc += 4

    def _exec_f(self, opcode: int, rd: int, imm16: int) -> None:
        """Execute format F instructions: opcode + rd + imm16 (LE)."""
        simm = imm16 if imm16 < 0x8000 else imm16 - 0x10000
        if opcode == MOVI16:
            self.registers[rd] = simm
            self.pc += 4
        elif opcode == ADDI16:
            self.registers[rd] = self._to_i32(self.registers[rd] + simm)
            self.pc += 4
        elif opcode == SUBI16:
            self.registers[rd] = self._to_i32(self.registers[rd] - simm)
            self.pc += 4
        else:
            self.pc += 4

    def _exec_g(self, opcode: int, rd: int, rs1: int, imm16: int) -> None:
        """Execute format G instructions: opcode + rd + rs1 + imm16 (LE)."""
        addr = self._to_i32(self.registers[rs1] + imm16)
        if opcode == LOADOFF:
            val = self._mem_read(self._to_u8(addr))
            self.registers[rd] = self._sign_extend_8(val)
            self.pc += 5
        elif opcode == STOREOFF:
            self._mem_write(self._to_u8(addr), self.registers[rd])
            self.pc += 5
        else:
            self.pc += 5

    def _exec_misc(self, opcode: int, rd: int) -> None:
        """Execute miscellaneous instructions."""
        if opcode == CALL:
            if self.pc + 2 >= len(self.bytecode):
                self.error = "Truncated CALL"
                self.halted = True
                return
            offset = self._sign_extend_8(self._read8(self.pc + 2))
            self._push(self.pc + 3)
            self.pc = self._to_u32(self.pc + 3 + offset)
        elif opcode == LOOP:
            if self.pc + 2 >= len(self.bytecode):
                self.error = "Truncated LOOP"
                self.halted = True
                return
            offset = self._sign_extend_8(self._read8(self.pc + 2))
            self.registers[rd] = self._to_i32(self.registers[rd] - 1)
            self.pc += 3
            if self.registers[rd] > 0:
                self.pc = self._to_u32(self.pc + offset)
        elif opcode == ENTER:
            # ENTER rd: allocate rd local slots on stack
            for _ in range(rd):
                self._push(0)
            self.pc += 2
        elif opcode == LEAVE:
            # LEAVE rd: deallocate rd local slots
            self.sp = max(0, self.sp - rd)
            self.pc += 2
        elif opcode == COPY:
            # COPY rd: copy memory block (simplified)
            self.pc += 2
        elif opcode == FILL:
            # FILL rd: fill memory with register value (simplified)
            self.pc += 2
        elif opcode == LOADI:
            self.pc += 2
        elif opcode == STOREI:
            self.pc += 2
        else:
            self.pc += 2

    def reset(self) -> None:
        """Reset VM state for a new execution."""
        self.registers = [0] * self.MAX_REGISTERS
        self.memory = bytearray(self.MEMORY_SIZE)
        self.stack = [0] * self.STACK_SIZE
        self.sp = 0
        self.pc = 0
        self.halted = False
        self.error = None
        self.instructions_executed = 0
        self.cycles_used = 0
        self.memory_reads = 0
        self.memory_writes = 0
        self.peak_stack_depth = 0
        self.bytecode = b""


class MiniVMAdapter(VMAdapter):
    """VMAdapter implementation using the embedded MiniVM."""

    def __init__(self) -> None:
        self._vm = MiniVM()

    @property
    def name(self) -> str:
        return "MiniVM-Python"

    def execute(self, bytecode: bytes, timeout: float = 10.0) -> ProfileResult:
        """Execute bytecode using the MiniVM and return profiling results."""
        self._vm.reset()
        self._vm.bytecode = bytecode

        start = time.perf_counter_ns()
        self._vm.run(bytecode, timeout=timeout)
        end = time.perf_counter_ns()

        registers_final = {i: self._vm.registers[i] for i in range(MiniVM.MAX_REGISTERS)}
        error = self._vm.error

        return ProfileResult(
            benchmark_name="",
            vm_name=self.name,
            wall_time_ns=end - start,
            instructions_executed=self._vm.instructions_executed,
            cycles_used=self._vm.cycles_used,
            memory_reads=self._vm.memory_reads,
            memory_writes=self._vm.memory_writes,
            peak_stack_depth=self._vm.peak_stack_depth,
            registers_final=registers_final,
            result_correct=False,
            error=error,
        )
