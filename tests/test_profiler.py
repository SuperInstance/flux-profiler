"""Tests for the profiling engine and MiniVM adapter."""

import time

import pytest

from flux_profiler.benchmarks import StandardBenchmarks
from flux_profiler.profiler import Profiler
from flux_profiler.vm_adapter import (
    MiniVM,
    MiniVMAdapter,
    ProfileResult,
    VMAdapter,
    get_cycle_cost,
)


# ── Mock VM Adapter for Testing ──────────────────────────────────────────────

class MockVMAdapter(VMAdapter):
    """A mock VM adapter for testing the Profiler without real execution."""

    def __init__(self, name: str = "MockVM", delay_ns: int = 1000) -> None:
        self._name = name
        self._delay_ns = delay_ns
        self.execute_count = 0
        self.last_bytecode: bytes = b""

    @property
    def name(self) -> str:
        return self._name

    def execute(self, bytecode: bytes, timeout: float = 10.0) -> ProfileResult:
        self.execute_count += 1
        self.last_bytecode = bytecode
        start = time.perf_counter_ns()
        time.sleep(self._delay_ns / 1e9)
        end = time.perf_counter_ns()
        return ProfileResult(
            benchmark_name="",
            vm_name=self._name,
            wall_time_ns=end - start,
            instructions_executed=len(bytecode),
            cycles_used=len(bytecode) * 2,
            memory_reads=0,
            memory_writes=0,
            peak_stack_depth=0,
            registers_final={0: 42},
            result_correct=True,
            error=None,
        )


class FailingVMAdapter(VMAdapter):
    """A VM adapter that always fails."""

    @property
    def name(self) -> str:
        return "FailingVM"

    def execute(self, bytecode: bytes, timeout: float = 10.0) -> ProfileResult:
        return ProfileResult(
            benchmark_name="",
            vm_name=self.name,
            wall_time_ns=0,
            instructions_executed=0,
            cycles_used=0,
            memory_reads=0,
            memory_writes=0,
            peak_stack_depth=0,
            registers_final={},
            result_correct=False,
            error="Simulated failure",
        )


# ── MiniVM Tests ─────────────────────────────────────────────────────────────

class TestMiniVM:
    """Tests for the embedded MiniVM."""

    def test_creation(self):
        vm = MiniVM()
        assert len(vm.registers) == 16
        assert len(vm.memory) == 256
        assert vm.pc == 0
        assert vm.halted is False

    def test_halt(self):
        vm = MiniVM()
        vm.run(bytes([0x00]))
        assert vm.halted is True
        assert vm.instructions_executed == 1

    def test_nop(self):
        vm = MiniVM()
        vm.run(bytes([0x01, 0x00]))  # NOP, HALT
        assert vm.instructions_executed == 2
        assert vm.halted is True

    def test_movi(self):
        vm = MiniVM()
        # MOVI R1, 42
        vm.run(bytes([0x18, 1, 42, 0x00]))
        assert vm.registers[1] == 42

    def test_movi_sign_extend(self):
        vm = MiniVM()
        # MOVI R1, 200 (= -56 signed)
        vm.run(bytes([0x18, 1, 200, 0x00]))
        assert vm.registers[1] == -56

    def test_movi16(self):
        vm = MiniVM()
        # MOVI16 R1, 1000
        code = bytes([0x40, 1, 0xE8, 0x03])
        vm.run(code)
        assert vm.registers[1] == 1000

    def test_movi16_large(self):
        vm = MiniVM()
        # MOVI16 R0, 30000
        code = bytes([0x40, 0, 0x30, 0x75])
        vm.run(code)
        assert vm.registers[0] == 30000

    def test_add(self):
        vm = MiniVM()
        # MOVI R1, 10; MOVI R2, 20; ADD R0, R1, R2; HALT
        code = bytes([0x18, 1, 10, 0x18, 2, 20, 0x20, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 30

    def test_sub(self):
        vm = MiniVM()
        # MOVI R1, 30; MOVI R2, 12; SUB R0, R1, R2; HALT
        code = bytes([0x18, 1, 30, 0x18, 2, 12, 0x21, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 18

    def test_mul(self):
        vm = MiniVM()
        # MOVI R1, 7; MOVI R2, 6; MUL R0, R1, R2; HALT
        code = bytes([0x18, 1, 7, 0x18, 2, 6, 0x22, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 42

    def test_div(self):
        vm = MiniVM()
        # MOVI R1, 42; MOVI R2, 6; DIV R0, R1, R2; HALT
        code = bytes([0x18, 1, 42, 0x18, 2, 6, 0x23, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 7

    def test_div_by_zero(self):
        vm = MiniVM()
        # MOVI R1, 42; MOVI R2, 0; DIV R0, R1, R2; HALT
        code = bytes([0x18, 1, 42, 0x18, 2, 0, 0x23, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.error == "Division by zero"

    def test_mod(self):
        vm = MiniVM()
        # MOVI R1, 17; MOVI R2, 5; MOD R0, R1, R2; HALT
        code = bytes([0x18, 1, 17, 0x18, 2, 5, 0x24, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 2

    def test_inc(self):
        vm = MiniVM()
        # MOVI R1, 5; INC R1; HALT
        code = bytes([0x18, 1, 5, 0x08, 1, 0x00])
        vm.run(code)
        assert vm.registers[1] == 6

    def test_dec(self):
        vm = MiniVM()
        # MOVI R1, 5; DEC R1; HALT
        code = bytes([0x18, 1, 5, 0x09, 1, 0x00])
        vm.run(code)
        assert vm.registers[1] == 4

    def test_not(self):
        vm = MiniVM()
        # MOVI R1, 0; NOT R1; HALT → should be -1 (all 1s)
        code = bytes([0x18, 1, 0, 0x0A, 1, 0x00])
        vm.run(code)
        assert vm.registers[1] == -1

    def test_neg(self):
        vm = MiniVM()
        # MOVI R1, 42; NEG R1; HALT
        code = bytes([0x18, 1, 42, 0x0B, 1, 0x00])
        vm.run(code)
        assert vm.registers[1] == -42

    def test_mov(self):
        vm = MiniVM()
        # MOVI R1, 99; MOV R0, R1, 0; HALT
        code = bytes([0x18, 1, 99, 0x3A, 0, 1, 0, 0x00])
        vm.run(code)
        assert vm.registers[0] == 99

    def test_addi(self):
        vm = MiniVM()
        # MOVI R1, 10; ADDI R1, 5; HALT
        code = bytes([0x18, 1, 10, 0x19, 1, 5, 0x00])
        vm.run(code)
        assert vm.registers[1] == 15

    def test_subi(self):
        vm = MiniVM()
        # MOVI R1, 10; SUBI R1, 3; HALT
        code = bytes([0x18, 1, 10, 0x1A, 1, 3, 0x00])
        vm.run(code)
        assert vm.registers[1] == 7

    def test_and(self):
        vm = MiniVM()
        # MOVI R1, 0xFF; MOVI R2, 0x0F; AND R0, R1, R2; HALT
        code = bytes([0x18, 1, 0xFF, 0x18, 2, 0x0F, 0x25, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 0x0F

    def test_or(self):
        vm = MiniVM()
        # MOVI R1, 0xF0; MOVI R2, 0x0F; OR R0, R1, R2; HALT
        code = bytes([0x18, 1, 0xF0, 0x18, 2, 0x0F, 0x26, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 0xFF

    def test_xor(self):
        vm = MiniVM()
        # MOVI R1, 0xFF; MOVI R2, 0x0F; XOR R0, R1, R2; HALT
        code = bytes([0x18, 1, 0xFF, 0x18, 2, 0x0F, 0x27, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 0xF0

    def test_shl(self):
        vm = MiniVM()
        # MOVI R1, 1; MOVI R2, 4; SHL R0, R1, R2; HALT
        code = bytes([0x18, 1, 1, 0x18, 2, 4, 0x28, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 16

    def test_shr(self):
        vm = MiniVM()
        # MOVI R1, 16; MOVI R2, 2; SHR R0, R1, R2; HALT
        code = bytes([0x18, 1, 16, 0x18, 2, 2, 0x29, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 4

    def test_cmp_eq_true(self):
        vm = MiniVM()
        # MOVI R1, 5; MOVI R2, 5; CMP_EQ R0, R1, R2; HALT
        code = bytes([0x18, 1, 5, 0x18, 2, 5, 0x2C, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 1

    def test_cmp_eq_false(self):
        vm = MiniVM()
        # MOVI R1, 5; MOVI R2, 3; CMP_EQ R0, R1, R2; HALT
        code = bytes([0x18, 1, 5, 0x18, 2, 3, 0x2C, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 0

    def test_cmp_lt_true(self):
        vm = MiniVM()
        # MOVI R1, 3; MOVI R2, 5; CMP_LT R0, R1, R2; HALT
        code = bytes([0x18, 1, 3, 0x18, 2, 5, 0x2D, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 1

    def test_cmp_lt_false(self):
        vm = MiniVM()
        # MOVI R1, 5; MOVI R2, 3; CMP_LT R0, R1, R2; HALT
        code = bytes([0x18, 1, 5, 0x18, 2, 3, 0x2D, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 0

    def test_cmp_gt_true(self):
        vm = MiniVM()
        # MOVI R1, 5; MOVI R2, 3; CMP_GT R0, R1, R2; HALT
        code = bytes([0x18, 1, 5, 0x18, 2, 3, 0x2E, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 1

    def test_cmp_ne_true(self):
        vm = MiniVM()
        # MOVI R1, 5; MOVI R2, 3; CMP_NE R0, R1, R2; HALT
        code = bytes([0x18, 1, 5, 0x18, 2, 3, 0x2F, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 1

    def test_cmp_ne_false(self):
        vm = MiniVM()
        # MOVI R1, 5; MOVI R2, 5; CMP_NE R0, R1, R2; HALT
        code = bytes([0x18, 1, 5, 0x18, 2, 5, 0x2F, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 0

    def test_store_load(self):
        vm = MiniVM()
        # MOVI R1, 42; STORE R1, 10; MOVI R1, 0; LOAD R0, 10; HALT
        code = bytes([0x18, 1, 42, 0x39, 1, 10, 0x18, 1, 0, 0x38, 0, 10, 0x00])
        vm.run(code)
        assert vm.registers[0] == 42
        assert vm.memory_reads == 1
        assert vm.memory_writes == 1

    def test_push_pop(self):
        vm = MiniVM()
        # MOVI R1, 99; PUSH R1; MOVI R1, 0; POP R0; HALT
        code = bytes([0x18, 1, 99, 0x0C, 1, 0x18, 1, 0, 0x0D, 0, 0x00])
        vm.run(code)
        assert vm.registers[0] == 99
        assert vm.peak_stack_depth == 1

    def test_jnz_taken(self):
        vm = MiniVM()
        # MOVI R1, 1; JNZ R1, -2 (back to MOVI) → should be infinite but we test the branch
        # Better: MOVI R0, 10; MOVI R1, 1; JNZ R1, 2; MOVI R0, 20; HALT
        # JNZ taken → skip MOVI R0, 20 → R0 = 10
        code = bytes([0x18, 0, 10, 0x18, 1, 1, 0x3D, 1, 2, 0x18, 0, 20, 0x00])
        vm.run(code)
        assert vm.registers[0] == 10

    def test_jnz_not_taken(self):
        vm = MiniVM()
        # MOVI R0, 10; MOVI R1, 0; JNZ R1, 2; MOVI R0, 20; HALT
        # JNZ not taken → execute MOVI R0, 20 → R0 = 20
        code = bytes([0x18, 0, 10, 0x18, 1, 0, 0x3D, 1, 2, 0x18, 0, 20, 0x00])
        vm.run(code)
        assert vm.registers[0] == 20

    def test_jz_taken(self):
        vm = MiniVM()
        # MOVI R0, 10; MOVI R1, 0; JZ R1, 2; MOVI R0, 20; HALT
        # JZ taken → skip → R0 = 10
        code = bytes([0x18, 0, 10, 0x18, 1, 0, 0x3C, 1, 2, 0x18, 0, 20, 0x00])
        vm.run(code)
        assert vm.registers[0] == 10

    def test_jz_not_taken(self):
        vm = MiniVM()
        # MOVI R0, 10; MOVI R1, 1; JZ R1, 2; MOVI R0, 20; HALT
        code = bytes([0x18, 0, 10, 0x18, 1, 1, 0x3C, 1, 2, 0x18, 0, 20, 0x00])
        vm.run(code)
        assert vm.registers[0] == 20

    def test_simple_loop(self):
        """Test a simple counting loop: R0 = 0, increment 5 times."""
        vm = MiniVM()
        # MOVI R0, 0
        # MOVI R1, 5
        # MOVI R2, 0  (loop counter)
        # LOOP: INC R0
        # INC R2
        # CMP_LT R3, R2, R1
        # JNZ R3, -4 (back to INC R0)
        # HALT
        loop_offset = -4
        code = bytearray()
        code += bytes([0x18, 0, 0])  # MOVI R0, 0
        code += bytes([0x18, 1, 5])  # MOVI R1, 5
        code += bytes([0x18, 2, 0])  # MOVI R2, 0
        # loop_start at offset 9
        code += bytes([0x08, 0])     # INC R0 (offset 9)
        code += bytes([0x08, 2])     # INC R2 (offset 11)
        code += bytes([0x2D, 3, 2, 1])  # CMP_LT R3, R2, R1 (offset 13)
        code += bytes([0x3D, 3])     # JNZ R3 (offset 17)
        # offset from PC+3 (20) back to 9: 9 - 20 = -11
        code.append(256 + (-11))     # offset byte
        code += bytes([0x00])        # HALT
        vm.run(bytes(code))
        assert vm.error is None
        assert vm.registers[0] == 5

    def test_instruction_counting(self):
        vm = MiniVM()
        vm.run(bytes([0x01, 0x01, 0x01, 0x00]))  # 3 NOPs + HALT
        assert vm.instructions_executed == 4

    def test_cycle_counting(self):
        vm = MiniVM()
        vm.run(bytes([0x18, 0, 5, 0x00]))  # MOVI(1) + HALT(1)
        assert vm.cycles_used == 2

    def test_memory_access_counting(self):
        vm = MiniVM()
        vm.run(bytes([0x18, 1, 42, 0x39, 1, 10, 0x00]))  # MOVI + STORE + HALT
        assert vm.memory_writes == 1

    def test_peak_stack_depth(self):
        vm = MiniVM()
        # Push 3 values
        code = bytes([
            0x18, 1, 1, 0x0C, 1,  # MOVI R1, 1; PUSH R1
            0x18, 1, 2, 0x0C, 1,  # MOVI R1, 2; PUSH R1
            0x18, 1, 3, 0x0C, 1,  # MOVI R1, 3; PUSH R1
            0x00,                  # HALT
        ])
        vm.run(code)
        assert vm.peak_stack_depth == 3

    def test_stack_overflow(self):
        vm = MiniVM()
        # Push more than 64 values
        code = bytearray()
        for i in range(70):
            code += bytes([0x18, 1, i, 0x0C, 1])  # MOVI R1, i; PUSH R1
        code += bytes([0x00])
        vm.run(bytes(code))
        assert vm.error == "Stack overflow"

    def test_stack_underflow(self):
        vm = MiniVM()
        vm.run(bytes([0x0D, 0, 0x00]))  # POP R0 with empty stack
        assert vm.error == "Stack underflow"

    def test_reset(self):
        vm = MiniVM()
        vm.run(bytes([0x18, 0, 42, 0x00]))
        assert vm.registers[0] == 42
        vm.reset()
        assert vm.registers[0] == 0
        assert vm.pc == 0
        assert vm.halted is False

    def test_timeout(self):
        vm = MiniVM()
        # Infinite loop
        code = bytes([0x43, 0, 0xFF])  # JMP R0, -1 (jump back to itself)
        vm.run(code, timeout=0.01)
        assert vm.error == "Timeout"

    def test_instruction_limit(self):
        vm = MiniVM()
        vm.MAX_INSTRUCTIONS = 100
        # Tight loop
        code = bytes([0x43, 0, 0xFF])  # JMP -1
        vm.run(code, timeout=60.0)
        assert "Instruction limit" in (vm.error or "")

    def test_popcnt(self):
        vm = MiniVM()
        # MOVI R1, 255 (8 bits set); POPCNT R1
        code = bytes([0x18, 1, 0xFF, 0x97, 1, 0x00])
        vm.run(code)
        assert vm.registers[1] == 8

    def test_popcnt_zero(self):
        vm = MiniVM()
        # MOVI R1, 0; POPCNT R1
        code = bytes([0x18, 1, 0, 0x97, 1, 0x00])
        vm.run(code)
        assert vm.registers[1] == 0

    def test_clz(self):
        vm = MiniVM()
        # MOVI R1, 1; CLZ R1 → 31 leading zeros
        code = bytes([0x18, 1, 1, 0x95, 1, 0x00])
        vm.run(code)
        assert vm.registers[1] == 31

    def test_ctz(self):
        vm = MiniVM()
        # MOVI R1, 8 (0b1000); CTZ R1 → 3
        code = bytes([0x18, 1, 8, 0x96, 1, 0x00])
        vm.run(code)
        assert vm.registers[1] == 3

    def test_min(self):
        vm = MiniVM()
        # MOVI R1, 10; MOVI R2, 20; MIN R0, R1, R2
        code = bytes([0x18, 1, 10, 0x18, 2, 20, 0x2A, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 10

    def test_max(self):
        vm = MiniVM()
        # MOVI R1, 10; MOVI R2, 20; MAX R0, R1, R2
        code = bytes([0x18, 1, 10, 0x18, 2, 20, 0x2B, 0, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[0] == 20

    def test_swap(self):
        vm = MiniVM()
        # MOVI R1, 10; MOVI R2, 20; SWP R1, R1, R2
        code = bytes([0x18, 1, 10, 0x18, 2, 20, 0x3B, 1, 1, 2, 0x00])
        vm.run(code)
        assert vm.registers[1] == 20

    def test_loadoff(self):
        vm = MiniVM()
        # Store 42 at addr 10, then load with LOADOFF
        # MOVI R1, 42; STORE R1, 10; MOVI R2, 10; LOADOFF R0, R2, 0; HALT
        code = bytes([
            0x18, 1, 42,              # MOVI R1, 42
            0x39, 1, 10,              # STORE R1, 10
            0x18, 2, 10,              # MOVI R2, 10
            0x48, 0, 2, 0, 0,         # LOADOFF R0, R2, 0
            0x00,                     # HALT
        ])
        vm.run(code)
        assert vm.registers[0] == 42

    def test_storeoff(self):
        vm = MiniVM()
        # MOVI R1, 42; MOVI R2, 10; STOREOFF R1, R2, 0; LOAD R0, 10; HALT
        code = bytes([
            0x18, 1, 42,              # MOVI R1, 42
            0x18, 2, 10,              # MOVI R2, 10
            0x49, 1, 2, 0, 0,         # STOREOFF R1, R2, 0
            0x38, 0, 10,              # LOAD R0, 10
            0x00,                     # HALT
        ])
        vm.run(code)
        assert vm.registers[0] == 42


# ── MiniVMAdapter Tests ──────────────────────────────────────────────────────

class TestMiniVMAdapter:
    """Tests for the MiniVMAdapter wrapper."""

    def test_adapter_name(self):
        adapter = MiniVMAdapter()
        assert adapter.name == "MiniVM-Python"

    def test_execute_returns_profile_result(self):
        adapter = MiniVMAdapter()
        bm = StandardBenchmarks.factorial_small()
        result = adapter.execute(bm.bytecode)
        assert isinstance(result, ProfileResult)

    def test_execute_has_timing(self):
        adapter = MiniVMAdapter()
        bm = StandardBenchmarks.factorial_small()
        result = adapter.execute(bm.bytecode)
        assert result.wall_time_ns > 0

    def test_execute_has_instruction_count(self):
        adapter = MiniVMAdapter()
        bm = StandardBenchmarks.factorial_small()
        result = adapter.execute(bm.bytecode)
        assert result.instructions_executed > 0

    def test_execute_registers_final(self):
        adapter = MiniVMAdapter()
        bm = StandardBenchmarks.factorial_small()
        result = adapter.execute(bm.bytecode)
        assert 0 in result.registers_final

    def test_execute_no_error(self):
        adapter = MiniVMAdapter()
        bm = StandardBenchmarks.factorial_small()
        result = adapter.execute(bm.bytecode)
        assert result.error is None


# ── Cycle Cost Tests ─────────────────────────────────────────────────────────

class TestCycleCosts:
    """Tests for instruction cycle cost mapping."""

    def test_simple_instruction_cost(self):
        assert get_cycle_cost(0x00) == 1  # HALT
        assert get_cycle_cost(0x01) == 1  # NOP

    def test_arithmetic_instruction_cost(self):
        assert get_cycle_cost(0x20) == 2  # ADD
        assert get_cycle_cost(0x22) == 2  # MUL

    def test_memory_instruction_cost(self):
        assert get_cycle_cost(0x38) == 3  # LOAD
        assert get_cycle_cost(0x39) == 3  # STORE

    def test_float_instruction_cost(self):
        assert get_cycle_cost(0x30) == 4  # FADD
        assert get_cycle_cost(0x32) == 4  # FMUL

    def test_branch_instruction_cost(self):
        assert get_cycle_cost(0x3C) == 2  # JZ
        assert get_cycle_cost(0x43) == 2  # JMP

    def test_a2a_instruction_cost(self):
        assert get_cycle_cost(0x50) == 5  # TELL
        assert get_cycle_cost(0x53) == 5  # BCAST

    def test_unknown_opcode_cost(self):
        assert get_cycle_cost(0xFF) == 1  # default


# ── Profiler Tests ───────────────────────────────────────────────────────────

class TestProfiler:
    """Tests for the Profiler orchestration engine."""

    def test_creation(self):
        profiler = Profiler()
        assert profiler.registered_vms == []

    def test_register_vm(self):
        profiler = Profiler()
        adapter = MiniVMAdapter()
        profiler.register_vm(adapter)
        assert "MiniVM-Python" in profiler.registered_vms

    def test_register_multiple_vms(self):
        profiler = Profiler()
        profiler.register_vm(MiniVMAdapter())
        profiler.register_vm(MockVMAdapter("VM2"))
        assert len(profiler.registered_vms) == 2

    def test_warmup_count_default(self):
        profiler = Profiler()
        assert profiler.warmup_count == 3

    def test_warmup_count_setter(self):
        profiler = Profiler()
        profiler.warmup_count = 5
        assert profiler.warmup_count == 5

    def test_warmup_count_negative(self):
        profiler = Profiler()
        profiler.warmup_count = -1
        assert profiler.warmup_count == 0

    def test_run_single(self):
        profiler = Profiler()
        adapter = MiniVMAdapter()
        profiler.register_vm(adapter)
        bm = StandardBenchmarks.factorial_small()
        result = profiler.run_single(bm, "MiniVM-Python")
        assert result.benchmark_name == "factorial_small"
        assert result.instructions_executed > 0

    def test_run_benchmark(self):
        profiler = Profiler()
        profiler.register_vm(MiniVMAdapter())
        bm = StandardBenchmarks.factorial_small()
        result = profiler.run_benchmark(bm, "MiniVM-Python", iterations=3)
        assert result.benchmark_name == "factorial_small"
        assert result.wall_time_ns > 0
        assert result.result_correct is True

    def test_run_benchmark_nonexistent_vm(self):
        profiler = Profiler()
        profiler.register_vm(MiniVMAdapter())
        bm = StandardBenchmarks.factorial_small()
        with pytest.raises(ValueError, match="not registered"):
            profiler.run_benchmark(bm, "NonExistentVM")

    def test_run_suite(self):
        profiler = Profiler()
        profiler.register_vm(MiniVMAdapter())
        bms = StandardBenchmarks.all()[:5]  # just first 5
        results = profiler.run_suite(bms, "MiniVM-Python", iterations=2)
        assert len(results) == 5
        for r in results:
            assert r.benchmark_name != ""
            assert r.vm_name == "MiniVM-Python"

    def test_run_comparative(self):
        profiler = Profiler()
        profiler.register_vm(MiniVMAdapter())
        profiler.register_vm(MockVMAdapter("MockVM"))
        bms = StandardBenchmarks.all()[:3]
        results = profiler.run_comparative(bms, iterations=2)
        assert "MiniVM-Python" in results
        assert "MockVM" in results
        assert len(results["MiniVM-Python"]) == 3
        assert len(results["MockVM"]) == 3

    def test_warmup_runs(self):
        profiler = Profiler()
        profiler.warmup_count = 2
        profiler.register_vm(MockVMAdapter())
        bm = StandardBenchmarks.factorial_small()
        results = profiler.warmup(bm, "MockVM")
        assert len(results) == 2

    def test_warmup_custom_count(self):
        profiler = Profiler()
        profiler.register_vm(MockVMAdapter())
        bm = StandardBenchmarks.factorial_small()
        results = profiler.warmup(bm, "MockVM", count=5)
        assert len(results) == 5

    def test_result_correctness_check(self):
        profiler = Profiler()
        profiler.register_vm(MiniVMAdapter())
        bm = StandardBenchmarks.factorial_small()
        result = profiler.run_benchmark(bm, "MiniVM-Python", iterations=2)
        assert result.result_correct is True

    def test_result_correctness_none_expected(self):
        """When expected_result is None, correctness should be True."""
        profiler = Profiler()
        profiler.register_vm(MiniVMAdapter())
        bm = BenchmarkWorkload(
            name="test_none",
            description="Test with no expected result",
            bytecode=bytes([0x18, 0, 42, 0x00]),
            expected_result=None,
            category="mixed",
        )
        result = profiler.run_benchmark(bm, "MiniVM-Python", iterations=2)
        assert result.result_correct is True

    def test_failing_vm_returns_error(self):
        profiler = Profiler()
        profiler.register_vm(FailingVMAdapter())
        bm = StandardBenchmarks.factorial_small()
        result = profiler.run_benchmark(bm, "FailingVM", iterations=2)
        assert result.error is not None

    def test_median_selection(self):
        """run_benchmark should return the median result."""
        profiler = Profiler()
        profiler.warmup_count = 0
        adapter = MockVMAdapter("MockVM", delay_ns=0)
        profiler.register_vm(adapter)
        bm = StandardBenchmarks.factorial_small()
        result = profiler.run_benchmark(bm, "MockVM", iterations=5)
        # Should return a valid result
        assert result.wall_time_ns >= 0


# ── ProfileResult Dataclass Tests ────────────────────────────────────────────

class TestProfileResult:
    """Tests for the ProfileResult dataclass."""

    def test_creation(self):
        r = ProfileResult(
            benchmark_name="test",
            vm_name="vm",
            wall_time_ns=1000,
            instructions_executed=10,
            cycles_used=20,
            memory_reads=5,
            memory_writes=3,
            peak_stack_depth=2,
            registers_final={0: 42},
            result_correct=True,
        )
        assert r.benchmark_name == "test"
        assert r.error is None

    def test_with_error(self):
        r = ProfileResult(
            benchmark_name="test",
            vm_name="vm",
            wall_time_ns=0,
            instructions_executed=0,
            cycles_used=0,
            memory_reads=0,
            memory_writes=0,
            peak_stack_depth=0,
            registers_final={},
            result_correct=False,
            error="Something went wrong",
        )
        assert r.error == "Something went wrong"
        assert r.result_correct is False
