"""Enhanced pytest tests for flux-profiler — new feature coverage."""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiler import (
    FluxProfiler, ProfileReport, OpcodeProfile, HotPath, RegisterUsage,
    MemoryAllocation, CallGraphNode, InstructionTiming,
    compare_profiles, OP_NAMES,
)


# ── Instruction-level per-opcode timing ─────────────────────────────────────

class TestInstructionLevelTiming:
    def test_wallclock_disabled_by_default(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        # Wall-clock disabled → instruction_timings should be empty
        assert len(report.instruction_timings) == 0

    def test_wallclock_enabled(self):
        p = FluxProfiler([0x18, 0, 42, 0x00], enable_wallclock=True)
        report = p.profile()
        # When enabled, should have some timings
        assert len(report.instruction_timings) > 0

    def test_wallclock_timings_have_ns_fields(self):
        p = FluxProfiler([0x18, 0, 42, 0x00], enable_wallclock=True)
        report = p.profile()
        for it in report.instruction_timings:
            assert it.total_ns >= 0
            assert it.avg_ns >= 0
            assert it.min_ns >= 0
            assert it.max_ns >= 0
            assert it.max_ns >= it.min_ns

    def test_wallclock_sorted_by_total(self):
        p = FluxProfiler([0x18, 0, 42, 0x18, 1, 10, 0x20, 2, 0, 1, 0x00], enable_wallclock=True)
        report = p.profile()
        totals = [it.total_ns for it in report.instruction_timings]
        assert totals == sorted(totals, reverse=True)

    def test_opcode_profiles_have_ns_fields(self):
        p = FluxProfiler([0x18, 0, 42, 0x00], enable_wallclock=True)
        report = p.profile()
        for op in report.opcode_profiles:
            assert hasattr(op, 'total_ns')
            assert hasattr(op, 'avg_ns')


# ── Function call graph profiling ──────────────────────────────────────────

class TestCallGraphProfiling:
    def test_no_call_graph_for_simple_program(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        assert len(report.call_graph) == 0

    def test_call_graph_records_edges(self):
        # CALL to PC 10, which has HALT
        bc = [0x18, 0, 42] + [0x01] * 6 + [0x48, 10] + [0x01] * 0 + [0x00] + [0x01] * 1 + [0x00]
        # Fix: bytecode with CALL at PC 9, callee at PC 10
        bc = bytearray()
        bc.extend([0x18, 0, 42])  # PC 0: MOVI R0, 42
        bc.extend([0x01] * 6)     # PC 3-8: NOPs
        bc.append(0x48)           # PC 9: CALL
        bc.append(11)             # PC 10: target = 11
        bc.append(0x00)           # PC 11: HALT (callee)
        bc.append(0x00)           # PC 12: HALT (return)
        p = FluxProfiler(list(bc))
        report = p.profile()
        assert len(report.call_graph) > 0

    def test_call_graph_has_callee_name(self):
        bc = bytearray()
        bc.extend([0x18, 0, 1])
        bc.append(0x48)  # CALL
        bc.append(5)     # target
        bc.extend([0x01] * 2)  # padding
        bc.append(0x00)  # HALT at callee
        bc.append(0x00)  # HALT after return
        p = FluxProfiler(list(bc))
        p.add_label("my_func", 5)
        report = p.profile()
        for cg in report.call_graph:
            assert isinstance(cg.callee_name, str)
            assert len(cg.callee_name) > 0

    def test_call_graph_caller_and_callee_pc(self):
        bc = bytearray()
        bc.extend([0x18, 0, 1])
        bc.append(0x48)
        bc.append(5)
        bc.extend([0x01] * 2)
        bc.append(0x00)
        bc.append(0x00)
        p = FluxProfiler(list(bc))
        report = p.profile()
        for cg in report.call_graph:
            assert cg.caller_pc >= 0
            assert cg.callee_pc >= 0

    def test_call_graph_count(self):
        bc = bytearray()
        bc.extend([0x18, 0, 2])  # R0 = 2
        loop_start = len(bc)
        bc.append(0x48)  # CALL
        bc.append(10)    # target
        bc.append(0x09)  # DEC R0
        bc.extend([0x3D, 0, 0xFD, 0x00])  # JNZ back (manual offset)
        # Fix the JNZ offset
        jnz_pos = len(bc) - 3
        bc[jnz_pos + 1] = 0  # reg
        back_offset = loop_start - (jnz_pos + 1)
        bc[jnz_pos + 2] = back_offset & 0xFF
        bc.append(0x00)  # skip
        # callee at PC 10
        while len(bc) < 10:
            bc.append(0x01)
        bc[10] = 0x00  # HALT
        bc.append(0x00)  # HALT after return
        p = FluxProfiler(list(bc))
        report = p.profile()
        # The loop calls multiple times
        assert any(cg.call_count >= 1 for cg in report.call_graph)

    def test_json_includes_call_graph(self):
        bc = bytearray()
        bc.extend([0x18, 0, 1])
        bc.append(0x48)
        bc.append(5)
        bc.extend([0x01] * 2)
        bc.append(0x00)
        bc.append(0x00)
        p = FluxProfiler(list(bc))
        report = p.profile()
        data = json.loads(report.to_json())
        assert "call_graph" in data


# ── Memory allocation tracking per instruction ─────────────────────────────

class TestMemoryAllocationTracking:
    def test_push_tracked(self):
        bc = [0x18, 0, 42, 0x0C, 0, 0x00]  # MOVI R0, 42; PUSH R0; HALT
        p = FluxProfiler(bc)
        report = p.profile()
        push_allocs = [m for m in report.memory_allocations if m.push_count > 0]
        assert len(push_allocs) >= 1

    def test_pop_tracked(self):
        bc = [0x18, 0, 42, 0x0C, 0, 0x0D, 1, 0x00]
        p = FluxProfiler(bc)
        report = p.profile()
        pop_allocs = [m for m in report.memory_allocations if m.pop_count > 0]
        assert len(pop_allocs) >= 1

    def test_net_bytes(self):
        bc = [0x18, 0, 42, 0x0C, 0, 0x0D, 1, 0x00]
        p = FluxProfiler(bc)
        report = p.profile()
        for ma in report.memory_allocations:
            assert ma.net_bytes == ma.push_count - ma.pop_count

    def test_memory_allocations_in_json(self):
        bc = [0x18, 0, 42, 0x0C, 0, 0x0D, 1, 0x00]
        p = FluxProfiler(bc)
        report = p.profile()
        data = json.loads(report.to_json())
        assert "memory_allocations" in data
        assert isinstance(data["memory_allocations"], list)

    def test_memory_in_markdown(self):
        bc = [0x18, 0, 42, 0x0C, 0, 0x0D, 1, 0x00]
        p = FluxProfiler(bc)
        report = p.profile()
        md = report.to_markdown()
        assert "Memory Allocations" in md

    def test_no_memory_for_simple_program(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        assert len(report.memory_allocations) == 0


# ── Hot-path detection enhancements ────────────────────────────────────────

class TestHotPathEnhancements:
    def test_custom_depth(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile(hot_path_depth=4)
        for hp in report.hot_paths:
            assert len(hp.sequence) == 4

    def test_depth_2(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile(hot_path_depth=2)
        for hp in report.hot_paths:
            assert len(hp.sequence) == 2

    def test_depth_clamped_to_max_10(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile(hot_path_depth=100)
        for hp in report.hot_paths:
            assert len(hp.sequence) <= 10


# ── Flame graph data generation ────────────────────────────────────────────

class TestFlameGraphGeneration:
    def test_no_flame_without_sampling(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        assert report.flame_graph["samples"] == 0

    def test_flame_with_sampling(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile(flame_sample_interval=3)
        assert report.flame_graph["samples"] > 0

    def test_flame_graph_format(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile(flame_sample_interval=2)
        fg = report.flame_graph
        assert fg["format"] == "flamegraph"
        assert isinstance(fg["roots"], list)
        assert isinstance(fg["samples"], int)

    def test_flame_graph_json_serializable(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile(flame_sample_interval=2)
        j = report.to_json()
        data = json.loads(j)
        assert "flame_graph" in data
        assert data["flame_graph"]["format"] == "flamegraph"

    def test_flamegraph_folded_format(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        p.profile(flame_sample_interval=3)
        folded = p.get_flamegraph_folded()
        assert isinstance(folded, str)
        lines = folded.strip().split("\n")
        # Each line is a stack sample; may be a single frame (no semicolons)
        assert len(lines) > 0
        for line in lines:
            assert len(line) > 0


# ── Profile comparison ─────────────────────────────────────────────────────

class TestProfileComparison:
    def test_compare_basic(self):
        p1 = FluxProfiler([0x18, 0, 42, 0x00])
        p2 = FluxProfiler([0x18, 0, 42, 0x00])
        r1 = p1.profile()
        r2 = p2.profile()
        diff = compare_profiles(r1, r2)
        assert "summary" in diff
        assert "opcode_diffs" in diff

    def test_compare_identical(self):
        p1 = FluxProfiler([0x18, 0, 42, 0x00])
        r1 = p1.profile()
        diff = compare_profiles(r1, r1)
        assert diff["summary"]["instruction_delta"] == 0
        assert diff["summary"]["cycle_delta"] == 0
        assert diff["summary"]["speedup"] == 1.0

    def test_compare_different_programs(self):
        p_before = FluxProfiler([0x18, 0, 10, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        p_after = FluxProfiler([0x18, 0, 10, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x00])  # simpler
        r1 = p_before.profile()
        r2 = p_after.profile()
        diff = compare_profiles(r1, r2)
        assert diff["summary"]["instruction_delta"] != 0

    def test_compare_opcode_diffs(self):
        p1 = FluxProfiler([0x18, 0, 10, 0x00])
        p2 = FluxProfiler([0x18, 0, 10, 0x18, 1, 20, 0x00])
        r1 = p1.profile()
        r2 = p2.profile()
        diff = compare_profiles(r1, r2)
        assert len(diff["opcode_diffs"]) > 0
        # Each diff should have the expected fields
        for od in diff["opcode_diffs"]:
            assert "name" in od
            assert "before_count" in od
            assert "after_count" in od
            assert "count_delta" in od
            assert "count_delta_pct" in od

    def test_compare_speedup(self):
        p1 = FluxProfiler([0x01] * 100 + [0x00])
        p2 = FluxProfiler([0x01] * 50 + [0x00])
        r1 = p1.profile()
        r2 = p2.profile()
        diff = compare_profiles(r1, r2)
        assert diff["summary"]["speedup"] > 1.0

    def test_compare_json_serializable(self):
        p1 = FluxProfiler([0x18, 0, 42, 0x00])
        p2 = FluxProfiler([0x18, 0, 42, 0x00])
        diff = compare_profiles(p1.profile(), p2.profile())
        j = json.dumps(diff)
        data = json.loads(j)
        assert "summary" in data


# ── Labels ─────────────────────────────────────────────────────────────────

class TestLabels:
    def test_add_label(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        p.add_label("start", 0)
        p.add_label("end", 3)
        report = p.profile()
        assert report.labels["start"] == 0
        assert report.labels["end"] == 3

    def test_labels_in_markdown(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        p.add_label("init", 0)
        report = p.profile()
        md = report.to_markdown()
        assert "init" in md

    def test_labels_in_json(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        p.add_label("init", 0)
        report = p.profile()
        data = json.loads(report.to_json())
        assert "labels" in data
        assert data["labels"]["init"] == 0


# ── ProfileReport data class ───────────────────────────────────────────────

class TestProfileReportEnhanced:
    def test_to_dict(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "total_cycles" in d
        assert "memory_allocations" in d
        assert "call_graph" in d
        assert "instruction_timings" in d
        assert "flame_graph" in d
        assert "labels" in d

    def test_empty_memory_and_call_graph(self):
        p = FluxProfiler([0x00])
        report = p.profile()
        assert report.memory_allocations == []
        assert report.call_graph == []

    def test_markdown_has_cycles_column(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "Cycles" in md
