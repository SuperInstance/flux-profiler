"""Tests for benchmark definitions.

Validates that all standard benchmarks generate valid bytecode,
produce correct results, and cover all expected categories.
"""

import math
import struct

import pytest

from flux_profiler.benchmarks import (
    BenchmarkWorkload,
    StandardBenchmarks,
    ABS, ADD, ADDI, ADDI16, AND, ANDI,
    ASK, BCAST, CALL, CLZ, CMP_EQ, CMP_GT, CMP_LT, CMP_NE, CTZ,
    DEC, DELEG, DIV, FADD, FDIV, FMUL, FSUB,
    FTOI, FILL, HALT, ITOF,
    INC, JGT, JMP, JLT, JNZ, JZ,
    LOAD, LOADOFF, LOOP,
    MIN, MOD, MOVI, MOVI16, MOV, MUL, NEG, NOP, NOT,
    OR, ORI, POP, POPCNT, PUSH,
    RET,
    SHL, SHLI, SHR, SHRI,
    STORE, STOREOFF, SUB, SUBI, SUBI16,
    SWP, TELL, XOR, XORI,
    encode_a, encode_b, encode_c, encode_d, encode_e, encode_f, encode_g,
)
from flux_profiler.vm_adapter import MiniVM


# ── Encoding Helper Tests ────────────────────────────────────────────────────

class TestEncodingHelpers:
    """Tests for bytecode encoding helper functions."""

    def test_encode_a_single_byte(self):
        result = encode_a(HALT)
        assert result == bytes([0x00])

    def test_encode_a_various_opcodes(self):
        for op in [NOP, HALT, RET]:
            result = encode_a(op)
            assert len(result) == 1
            assert result[0] == op

    def test_encode_b_format(self):
        result = encode_b(INC, 5)
        assert result == bytes([INC, 5])

    def test_encode_b_masks_register(self):
        result = encode_b(DEC, 0xFF)
        assert result == bytes([DEC, 0x0F])

    def test_encode_c_format(self):
        result = encode_c(0x03, 42)
        assert result == bytes([0x03, 42])

    def test_encode_c_masks_imm8(self):
        result = encode_c(0x03, 0x1FF)
        assert result == bytes([0x03, 0xFF])

    def test_encode_d_format(self):
        result = encode_d(MOVI, 3, 100)
        assert result == bytes([MOVI, 3, 100])

    def test_encode_d_masks_rd(self):
        result = encode_d(MOVI, 17, 5)
        assert result[1] == 1  # 17 & 0xF = 1

    def test_encode_d_masks_imm8(self):
        result = encode_d(MOVI, 0, 300)
        assert result[2] == 44  # 300 & 0xFF

    def test_encode_e_format(self):
        result = encode_e(ADD, 1, 2, 3)
        assert result == bytes([ADD, 1, 2, 3])

    def test_encode_e_masks_all_registers(self):
        result = encode_e(ADD, 0x1F, 0x2F, 0x3F)
        assert result == bytes([ADD, 0xF, 0xF, 0xF])

    def test_encode_f_format(self):
        result = encode_f(MOVI16, 2, 0x1234)
        assert result == bytes([MOVI16, 2, 0x34, 0x12])

    def test_encode_f_zero(self):
        result = encode_f(MOVI16, 0, 0)
        assert result == bytes([MOVI16, 0, 0, 0])

    def test_encode_f_max_imm16(self):
        result = encode_f(MOVI16, 0, 0xFFFF)
        assert result == bytes([MOVI16, 0, 0xFF, 0xFF])

    def test_encode_g_format(self):
        result = encode_g(LOADOFF, 1, 2, 0x5678)
        assert result == bytes([LOADOFF, 1, 2, 0x78, 0x56])

    def test_encode_g_masks_rd_rs1(self):
        result = encode_g(LOADOFF, 0x1F, 0x2F, 0)
        assert result[1] == 0xF
        assert result[2] == 0xF


# ── Opcode Constants Tests ───────────────────────────────────────────────────

class TestOpcodeConstants:
    """Verify opcode values are correct."""

    def test_basic_opcodes(self):
        assert HALT == 0x00
        assert NOP == 0x01
        assert RET == 0x02

    def test_unary_opcodes(self):
        assert INC == 0x08
        assert DEC == 0x09
        assert NOT == 0x0A
        assert NEG == 0x0B

    def test_stack_opcodes(self):
        assert PUSH == 0x0C
        assert POP == 0x0D

    def test_immediate_arithmetic_opcodes(self):
        assert MOVI == 0x18
        assert ADDI == 0x19
        assert SUBI == 0x1A

    def test_register_arithmetic_opcodes(self):
        assert ADD == 0x20
        assert SUB == 0x21
        assert MUL == 0x22
        assert DIV == 0x23

    def test_comparison_opcodes(self):
        assert CMP_EQ == 0x2C
        assert CMP_LT == 0x2D
        assert CMP_GT == 0x2E
        assert CMP_NE == 0x2F

    def test_float_opcodes(self):
        assert FADD == 0x30
        assert FSUB == 0x31
        assert FMUL == 0x32
        assert FDIV == 0x33

    def test_memory_opcodes(self):
        assert LOAD == 0x38
        assert STORE == 0x39
        assert MOV == 0x3A
        assert LOADOFF == 0x48
        assert STOREOFF == 0x49

    def test_branch_opcodes(self):
        assert JZ == 0x3C
        assert JNZ == 0x3D
        assert JMP == 0x43

    def test_a2a_opcodes(self):
        assert TELL == 0x50
        assert ASK == 0x51
        assert BCAST == 0x53

    def test_special_opcodes(self):
        assert POPCNT == 0x97
        assert CLZ == 0x95
        assert CTZ == 0x96


# ── Benchmark Generation Tests ───────────────────────────────────────────────

class TestBenchmarkGeneration:
    """Tests that all benchmarks generate valid bytecode."""

    @pytest.fixture
    def vm(self):
        return MiniVM()

    def test_factorial_small_generates_bytecode(self):
        bm = StandardBenchmarks.factorial_small()
        assert isinstance(bm, BenchmarkWorkload)
        assert len(bm.bytecode) > 0
        assert bm.name == "factorial_small"
        assert bm.category == "arithmetic"

    def test_factorial_small_result(self, vm):
        bm = StandardBenchmarks.factorial_small()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 120

    def test_factorial_large_generates_bytecode(self):
        bm = StandardBenchmarks.factorial_large()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.name == "factorial_large"
        assert len(bm.bytecode) > 0

    def test_factorial_large_runs(self, vm):
        bm = StandardBenchmarks.factorial_large()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        # 12! = 479001600, may overflow i32
        assert vm.instructions_executed > 0

    def test_fibonacci_generates_bytecode(self):
        bm = StandardBenchmarks.fibonacci()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "control_flow"

    def test_fibonacci_result(self, vm):
        bm = StandardBenchmarks.fibonacci()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 610

    def test_matrix_multiply_generates_bytecode(self):
        bm = StandardBenchmarks.matrix_multiply()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "memory"

    def test_matrix_multiply_result(self, vm):
        bm = StandardBenchmarks.matrix_multiply()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 9

    def test_bubble_sort_generates_bytecode(self):
        bm = StandardBenchmarks.bubble_sort()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "memory"

    def test_bubble_sort_result(self, vm):
        bm = StandardBenchmarks.bubble_sort()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 36

    def test_stack_heavy_generates_bytecode(self):
        bm = StandardBenchmarks.stack_heavy()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "stack"

    def test_stack_heavy_result(self, vm):
        bm = StandardBenchmarks.stack_heavy()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 1275

    def test_branch_heavy_generates_bytecode(self):
        bm = StandardBenchmarks.branch_heavy()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "control_flow"

    def test_branch_heavy_result(self, vm):
        bm = StandardBenchmarks.branch_heavy()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 334

    def test_memory_bandwidth_generates_bytecode(self):
        bm = StandardBenchmarks.memory_bandwidth()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "memory"

    def test_memory_bandwidth_result(self, vm):
        bm = StandardBenchmarks.memory_bandwidth()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 2016

    def test_float_arithmetic_generates_bytecode(self):
        bm = StandardBenchmarks.float_arithmetic()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "float"

    def test_float_arithmetic_result(self, vm):
        bm = StandardBenchmarks.float_arithmetic()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 100

    def test_register_pressure_generates_bytecode(self):
        bm = StandardBenchmarks.register_pressure()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "mixed"

    def test_register_pressure_result(self, vm):
        bm = StandardBenchmarks.register_pressure()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 120

    def test_gcd_generates_bytecode(self):
        bm = StandardBenchmarks.gcd()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "control_flow"

    def test_gcd_result(self, vm):
        bm = StandardBenchmarks.gcd()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 6

    def test_prime_sieve_generates_bytecode(self):
        bm = StandardBenchmarks.prime_sieve()
        assert isinstance(bm, BenchmarkWorkload)

    def test_prime_sieve_result(self, vm):
        bm = StandardBenchmarks.prime_sieve()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 25

    def test_a2a_simulation_generates_bytecode(self):
        bm = StandardBenchmarks.a2a_simulation()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "a2a"

    def test_a2a_simulation_result(self, vm):
        bm = StandardBenchmarks.a2a_simulation()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 50

    def test_syscall_heavy_generates_bytecode(self):
        bm = StandardBenchmarks.syscall_heavy()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "mixed"

    def test_syscall_heavy_result(self, vm):
        bm = StandardBenchmarks.syscall_heavy()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 100

    def test_noop_baseline_generates_bytecode(self):
        bm = StandardBenchmarks.noop_baseline()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "mixed"
        # Should have 1000+ NOPs
        assert bm.bytecode.count(bytes([NOP])) >= 1000

    def test_noop_baseline_result(self, vm):
        bm = StandardBenchmarks.noop_baseline()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 42

    def test_memory_random_generates_bytecode(self):
        bm = StandardBenchmarks.memory_random()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "memory"

    def test_memory_random_result(self, vm):
        bm = StandardBenchmarks.memory_random()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == bm.expected_result

    def test_recursive_call_generates_bytecode(self):
        bm = StandardBenchmarks.recursive_call()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "stack"

    def test_recursive_call_result(self, vm):
        bm = StandardBenchmarks.recursive_call()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 55

    def test_bit_manipulation_generates_bytecode(self):
        bm = StandardBenchmarks.bit_manipulation()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "arithmetic"

    def test_bit_manipulation_result(self, vm):
        bm = StandardBenchmarks.bit_manipulation()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        expected = sum(bin(i).count('1') for i in range(64))
        assert vm.registers[0] == expected

    def test_comparison_sort_generates_bytecode(self):
        bm = StandardBenchmarks.comparison_sort()
        assert isinstance(bm, BenchmarkWorkload)

    def test_comparison_sort_result(self, vm):
        bm = StandardBenchmarks.comparison_sort()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 9

    def test_long_program_generates_bytecode(self):
        bm = StandardBenchmarks.long_program()
        assert isinstance(bm, BenchmarkWorkload)
        assert bm.category == "mixed"
        assert len(bm.bytecode) >= 500

    def test_long_program_result(self, vm):
        bm = StandardBenchmarks.long_program()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.error is None
        assert vm.registers[0] == 5525

    def test_all_benchmarks_returns_list(self):
        benchmarks = StandardBenchmarks.all()
        assert isinstance(benchmarks, list)
        assert len(benchmarks) == 20

    def test_all_benchmarks_have_names(self):
        benchmarks = StandardBenchmarks.all()
        for bm in benchmarks:
            assert bm.name, f"Benchmark missing name"
            assert isinstance(bm.name, str)

    def test_all_benchmarks_have_descriptions(self):
        benchmarks = StandardBenchmarks.all()
        for bm in benchmarks:
            assert bm.description, f"{bm.name} missing description"

    def test_all_benchmarks_have_bytecode(self):
        benchmarks = StandardBenchmarks.all()
        for bm in benchmarks:
            assert len(bm.bytecode) > 0, f"{bm.name} has empty bytecode"

    def test_all_benchmarks_have_categories(self):
        benchmarks = StandardBenchmarks.all()
        for bm in benchmarks:
            assert bm.category in (
                "arithmetic", "control_flow", "memory", "stack",
                "mixed", "a2a", "float",
            ), f"{bm.name} has invalid category: {bm.category}"

    def test_all_benchmarks_have_estimated_cycles(self):
        benchmarks = StandardBenchmarks.all()
        for bm in benchmarks:
            assert bm.estimated_cycles > 0, f"{bm.name} has zero estimated cycles"

    def test_all_benchmarks_unique_names(self):
        benchmarks = StandardBenchmarks.all()
        names = [bm.name for bm in benchmarks]
        assert len(names) == len(set(names)), "Duplicate benchmark names found"


# ── Category Coverage Tests ──────────────────────────────────────────────────

class TestCategoryCoverage:
    """Ensure benchmarks cover all expected categories."""

    def test_all_categories_covered(self):
        benchmarks = StandardBenchmarks.all()
        categories = set(bm.category for bm in benchmarks)
        expected = {"arithmetic", "control_flow", "memory", "stack", "mixed", "a2a", "float"}
        assert expected.issubset(categories), f"Missing categories: {expected - categories}"

    def test_by_category_arithmetic(self):
        bms = StandardBenchmarks.by_category("arithmetic")
        assert len(bms) > 0
        for bm in bms:
            assert bm.category == "arithmetic"

    def test_by_category_control_flow(self):
        bms = StandardBenchmarks.by_category("control_flow")
        assert len(bms) > 0

    def test_by_category_memory(self):
        bms = StandardBenchmarks.by_category("memory")
        assert len(bms) > 0

    def test_by_category_stack(self):
        bms = StandardBenchmarks.by_category("stack")
        assert len(bms) > 0

    def test_by_category_mixed(self):
        bms = StandardBenchmarks.by_category("mixed")
        assert len(bms) > 0

    def test_by_category_a2a(self):
        bms = StandardBenchmarks.by_category("a2a")
        assert len(bms) > 0

    def test_by_category_float(self):
        bms = StandardBenchmarks.by_category("float")
        assert len(bms) > 0

    def test_by_category_nonexistent(self):
        bms = StandardBenchmarks.by_category("nonexistent")
        assert len(bms) == 0


# ── Bytecode Validity Tests ──────────────────────────────────────────────────

class TestBytecodeValidity:
    """Verify bytecode structure and validity."""

    @pytest.fixture
    def vm(self):
        return MiniVM()

    def test_all_benchmarks_execute_without_crash(self, vm):
        """Every benchmark should execute to HALT or known error."""
        benchmarks = StandardBenchmarks.all()
        for bm in benchmarks:
            vm.reset()
            vm.run(bm.bytecode, timeout=5.0)
            assert vm.halted, f"{bm.name} did not halt"

    def test_all_benchmarks_with_expected_results_match(self, vm):
        """Benchmarks with expected results should produce them."""
        benchmarks = StandardBenchmarks.all()
        correct = 0
        total = 0
        for bm in benchmarks:
            if bm.expected_result is not None:
                total += 1
                vm.reset()
                vm.run(bm.bytecode, timeout=5.0)
                if vm.error is None and vm.registers[0] == bm.expected_result:
                    correct += 1
        assert correct == total, f"Only {correct}/{total} benchmarks produced correct results"

    def test_bytecode_starts_with_valid_opcode(self):
        """Every benchmark's bytecode should start with a known opcode."""
        benchmarks = StandardBenchmarks.all()
        known_opcodes = {
            HALT, NOP, RET, INC, DEC, NOT, NEG, PUSH, POP,
            MOVI, ADDI, SUBI, ANDI, ORI, XORI, SHLI, SHRI,
            ADD, SUB, MUL, DIV, MOD, AND, OR, XOR, SHL, SHR,
            MOVI16, ADDI16, SUBI16,
            LOAD, STORE, MOV, FILL,
        }
        for bm in benchmarks:
            first_opcode = bm.bytecode[0]
            assert first_opcode in known_opcodes, (
                f"{bm.name} starts with unknown opcode 0x{first_opcode:02x}"
            )

    def test_noop_baseline_instruction_count(self, vm):
        """noop_baseline should execute ~1003 instructions (MOVI + 1000 NOPs + HALT)."""
        bm = StandardBenchmarks.noop_baseline()
        vm.reset()
        vm.run(bm.bytecode)
        assert vm.instructions_executed >= 1000


# ── BenchmarkWorkload Dataclass Tests ────────────────────────────────────────

class TestBenchmarkWorkload:
    """Tests for the BenchmarkWorkload dataclass."""

    def test_creation_with_required_fields(self):
        bm = BenchmarkWorkload(
            name="test",
            description="Test benchmark",
            bytecode=bytes([HALT]),
            expected_result=0,
            category="mixed",
        )
        assert bm.name == "test"
        assert bm.expected_result == 0

    def test_creation_with_none_expected_result(self):
        bm = BenchmarkWorkload(
            name="test",
            description="Test benchmark",
            bytecode=bytes([HALT]),
            expected_result=None,
            category="mixed",
        )
        assert bm.expected_result is None

    def test_default_estimated_cycles(self):
        bm = BenchmarkWorkload(
            name="test",
            description="Test",
            bytecode=bytes([HALT]),
            expected_result=None,
            category="mixed",
        )
        assert bm.estimated_cycles == 0

    def test_custom_estimated_cycles(self):
        bm = BenchmarkWorkload(
            name="test",
            description="Test",
            bytecode=bytes([HALT]),
            expected_result=None,
            category="mixed",
            estimated_cycles=500,
        )
        assert bm.estimated_cycles == 500
