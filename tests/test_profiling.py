"""Comprehensive pytest tests for FluxProfiler data collection."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiler import FluxProfiler, ProfileReport, OpcodeProfile, HotPath, RegisterUsage, OP_NAMES


# ── basic profiling ────────────────────────────────────

class TestBasicProfiling:
    def test_halt_only(self):
        p = FluxProfiler([0x00])
        report = p.profile()
        assert report.total_instructions == 1
        assert report.total_cycles == 1
        assert report.program_size == 1

    def test_empty_bytecode(self):
        p = FluxProfiler([])
        report = p.profile()
        assert report.total_instructions == 0
        assert report.total_cycles == 0
        assert report.program_size == 0

    def test_single_movi(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        assert report.total_instructions == 2  # MOVI + HALT

    def test_multiple_instructions(self):
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x00])
        report = p.profile()
        assert report.total_instructions == 3  # 2x MOVI + HALT

    def test_max_cycles_limit(self):
        """Profiler should respect max_cycles limit."""
        p = FluxProfiler([0x18, 0, 1, 0x3D, 0, 0, 0x00])  # infinite loop
        report = p.profile(max_cycles=50)
        assert report.total_instructions <= 50


# ── opcode counting ────────────────────────────────────

class TestOpcodeCounting:
    def test_single_opcode_count(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        op_map = {o.opcode: o for o in report.opcode_profiles}
        assert op_map[0x18].name == "MOVI"
        assert op_map[0x18].count == 1
        assert op_map[0x00].name == "HALT"
        assert op_map[0x00].count == 1

    def test_multiple_movi_counts(self):
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x18, 2, 30, 0x00])
        report = p.profile()
        op_map = {o.opcode: o for o in report.opcode_profiles}
        assert op_map[0x18].count == 3  # 3 MOVI instructions

    def test_add_counted(self):
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        report = p.profile()
        op_names = [o.name for o in report.opcode_profiles]
        assert "ADD" in op_names

    def test_nop_counted(self):
        p = FluxProfiler([0x01, 0x01, 0x01, 0x00])
        report = p.profile()
        op_map = {o.opcode: o for o in report.opcode_profiles}
        assert op_map[0x01].count == 3

    def test_inc_dec_counted(self):
        p = FluxProfiler([0x08, 0, 0x09, 0, 0x00])
        report = p.profile()
        op_map = {o.opcode: o for o in report.opcode_profiles}
        assert op_map[0x08].count == 1  # INC
        assert op_map[0x09].count == 1  # DEC


# ── percentage calculations ────────────────────────────

class TestPercentages:
    def test_percentages_sum_to_100(self):
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        report = p.profile()
        total_pct = sum(o.percentage for o in report.opcode_profiles)
        assert abs(total_pct - 100.0) < 0.01

    def test_single_opcode_100_percent(self):
        p = FluxProfiler([0x00])
        report = p.profile()
        assert len(report.opcode_profiles) == 1
        assert report.opcode_profiles[0].percentage == 100.0

    def test_equal_split(self):
        p = FluxProfiler([0x01, 0x00])
        report = p.profile()
        assert len(report.opcode_profiles) == 2
        for op in report.opcode_profiles:
            assert op.percentage == 50.0

    def test_unknown_opcode_named(self):
        """Opcodes not in OP_NAMES should be named as hex strings."""
        p = FluxProfiler([0xFE, 0x00])  # 0xFE is unknown
        report = p.profile()
        # Unknown opcode should still be counted
        unknown_ops = [o for o in report.opcode_profiles if o.opcode == 0xFE]
        assert len(unknown_ops) == 1
        assert "0xfe" in unknown_ops[0].name.lower()


# ── cycle estimation ───────────────────────────────────

class TestCycleEstimation:
    def test_halt_costs_1(self):
        assert FluxProfiler.CYCLE_COSTS[0x00] == 1

    def test_nop_costs_1(self):
        assert FluxProfiler.CYCLE_COSTS[0x01] == 1

    def test_movi_costs_1(self):
        assert FluxProfiler.CYCLE_COSTS[0x18] == 1

    def test_add_costs_2(self):
        assert FluxProfiler.CYCLE_COSTS[0x20] == 2

    def test_mul_costs_more_than_add(self):
        assert FluxProfiler.CYCLE_COSTS[0x22] > FluxProfiler.CYCLE_COSTS[0x20]

    def test_div_costs_more_than_mul(self):
        assert FluxProfiler.CYCLE_COSTS[0x23] >= FluxProfiler.CYCLE_COSTS[0x22]

    def test_total_cycles_accumulated(self):
        p = FluxProfiler([0x01, 0x01, 0x01, 0x00])
        report = p.profile()
        # 3 NOPs (1 each) + 1 HALT (1) = 4
        assert report.total_cycles == 4

    def test_mul_program_cycles(self):
        # MOVI(1) + MOVI(1) + MUL(3) + HALT(1) = 6
        p = FluxProfiler([0x18, 0, 6, 0x18, 1, 7, 0x22, 2, 0, 1, 0x00])
        report = p.profile()
        assert report.total_cycles == 6

    def test_push_costs_2(self):
        assert FluxProfiler.CYCLE_COSTS[0x0C] == 2

    def test_pop_costs_2(self):
        assert FluxProfiler.CYCLE_COSTS[0x0D] == 2


# ── IPC (instructions per cycle) ───────────────────────

class TestIPC:
    def test_ipc_positive(self):
        p = FluxProfiler([0x00])
        report = p.profile()
        assert report.ipc > 0

    def test_ipc_equals_1_for_uniform_cost(self):
        """All NOPs + HALT cost 1 cycle each, so IPC should be 1.0."""
        p = FluxProfiler([0x01, 0x01, 0x00])
        report = p.profile()
        assert report.ipc == 1.0

    def test_ipc_less_than_1_for_heavy_ops(self):
        """MUL costs 3 cycles but counts as 1 instruction → IPC < 1."""
        p = FluxProfiler([0x22, 0, 0, 0, 0x00])  # MUL + HALT
        report = p.profile()
        assert report.ipc < 1.0

    def test_ipc_zero_for_empty(self):
        p = FluxProfiler([])
        report = p.profile()
        assert report.ipc == 0.0


# ── hot path detection ─────────────────────────────────

class TestHotPathDetection:
    def test_no_hot_path_for_short_program(self):
        """Program with fewer than 3 instructions can't have 3-instruction sequences."""
        p = FluxProfiler([0x00])
        report = p.profile()
        assert len(report.hot_paths) == 0

    def test_hot_path_detected_in_loop(self):
        """Factorial loop should produce detectable hot paths."""
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        assert len(report.hot_paths) > 0

    def test_hot_path_sequence_content(self):
        """Hot paths should contain instruction names."""
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        for hp in report.hot_paths:
            assert len(hp.sequence) == 3
            assert all(isinstance(name, str) for name in hp.sequence)

    def test_hot_path_has_start_pc(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        for hp in report.hot_paths:
            assert hp.start_pc >= 0

    def test_hot_path_has_count(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        for hp in report.hot_paths:
            assert hp.count >= 1

    def test_hot_paths_limited_to_top_5(self):
        """Should return at most 5 hot paths."""
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        assert len(report.hot_paths) <= 5

    def test_hot_paths_sorted_by_count(self):
        """Hot paths should be sorted by count (most common first)."""
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        counts = [hp.count for hp in report.hot_paths]
        assert counts == sorted(counts, reverse=True)

    def test_linear_program_hot_paths(self):
        """Linear program should still have hot paths if 3+ instructions."""
        p = FluxProfiler([0x18, 0, 1, 0x18, 1, 2, 0x18, 2, 3, 0x00])
        report = p.profile()
        assert len(report.hot_paths) >= 1
        # Each sequence appears exactly once
        for hp in report.hot_paths:
            assert hp.count == 1


# ── register usage tracking ────────────────────────────

class TestRegisterUsage:
    def test_single_register_written(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        reg_map = {r.register: r for r in report.register_usage}
        assert 0 in reg_map
        assert reg_map[0].writes == 1

    def test_multiple_registers(self):
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        report = p.profile()
        regs_used = [r.register for r in report.register_usage]
        assert 0 in regs_used
        assert 1 in regs_used
        assert 2 in regs_used

    def test_add_reads_two_registers(self):
        """ADD R2, R0, R1 should read R0 and R1, write R2."""
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        report = p.profile()
        reg_map = {r.register: r for r in report.register_usage}
        assert reg_map[0].reads >= 1  # R0 read by ADD
        assert reg_map[1].reads >= 1  # R1 read by ADD
        assert reg_map[2].writes >= 1  # R2 written by ADD

    def test_inc_writes_register(self):
        p = FluxProfiler([0x08, 0, 0x00])
        report = p.profile()
        reg_map = {r.register: r for r in report.register_usage}
        assert reg_map[0].writes == 1

    def test_push_reads_register(self):
        p = FluxProfiler([0x18, 0, 42, 0x0C, 0, 0x00])
        report = p.profile()
        reg_map = {r.register: r for r in report.register_usage}
        assert reg_map[0].reads >= 1  # PUSH reads

    def test_pop_writes_register(self):
        p = FluxProfiler([0x18, 0, 42, 0x0C, 0, 0x0D, 1, 0x00])
        report = p.profile()
        reg_map = {r.register: r for r in report.register_usage}
        assert reg_map[1].writes >= 1  # POP writes

    def test_no_registers_for_halt(self):
        p = FluxProfiler([0x00])
        report = p.profile()
        assert len(report.register_usage) == 0

    def test_total_equals_reads_plus_writes(self):
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        report = p.profile()
        for ru in report.register_usage:
            assert ru.total == ru.reads + ru.writes

    def test_register_usage_sorted_by_register_number(self):
        p = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x20, 2, 0, 1, 0x00])
        report = p.profile()
        reg_numbers = [r.register for r in report.register_usage]
        # all_regs is sorted, so register_usage should be sorted by register number
        assert reg_numbers == sorted(reg_numbers)


# ── PC visit tracking ──────────────────────────────────

class TestPCVisits:
    def test_pc_visits_recorded(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        p.profile()
        assert p.pc_visits[0] >= 1  # first instruction visited
        assert p.pc_visits[3] >= 1  # HALT visited

    def test_loop_pc_visited_multiple_times(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        p.profile()
        # Some PCs in the loop should be visited multiple times
        visited_multiple = sum(1 for v in p.pc_visits.values() if v > 1)
        assert visited_multiple >= 2


# ── execution trace ────────────────────────────────────

class TestExecutionTrace:
    def test_trace_records_all(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        assert len(p.execution_trace) == 2  # MOVI + HALT

    def test_trace_tuples(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        p.profile()
        assert all(isinstance(t, tuple) and len(t) == 2 for t in p.execution_trace)

    def test_trace_pcs_match_instructions(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        p.profile()
        assert p.execution_trace[0] == (0, 0x18)  # PC 0, MOVI
        assert p.execution_trace[1] == (3, 0x00)  # PC 3, HALT


# ── data classes ───────────────────────────────────────

class TestDataClasses:
    def test_opcode_profile_fields(self):
        op = OpcodeProfile(opcode=0x18, name="MOVI", count=5, percentage=50.0, cycles_estimate=5)
        assert op.opcode == 0x18
        assert op.name == "MOVI"
        assert op.count == 5
        assert op.percentage == 50.0
        assert op.cycles_estimate == 5

    def test_hot_path_fields(self):
        hp = HotPath(sequence=["MOVI", "ADD", "HALT"], count=10, start_pc=0)
        assert hp.sequence == ["MOVI", "ADD", "HALT"]
        assert hp.count == 10
        assert hp.start_pc == 0

    def test_register_usage_fields(self):
        ru = RegisterUsage(register=0, reads=5, writes=3, total=8)
        assert ru.register == 0
        assert ru.reads == 5
        assert ru.writes == 3
        assert ru.total == 8


# ── OP_NAMES coverage ──────────────────────────────────

class TestOPNames:
    def test_all_known_opcodes_have_names(self):
        """Verify standard opcodes have human-readable names."""
        important_ops = [0x00, 0x01, 0x08, 0x09, 0x0C, 0x0D, 0x18, 0x19, 0x20, 0x21, 0x22, 0x23, 0x3A, 0x3C, 0x3D, 0x46]
        for op in important_ops:
            assert op in OP_NAMES, f"Missing name for opcode 0x{op:02x}"

    def test_names_are_strings(self):
        for op, name in OP_NAMES.items():
            assert isinstance(name, str)


# ── signed byte helper ─────────────────────────────────

class TestSignedByte:
    def test_positive_byte(self):
        p = FluxProfiler([0x00])
        assert p._signed_byte(0) == 0
        assert p._signed_byte(127) == 127

    def test_negative_byte(self):
        p = FluxProfiler([0x00])
        assert p._signed_byte(128) == -128
        assert p._signed_byte(255) == -1
        assert p._signed_byte(200) == -56
