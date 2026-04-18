"""Comprehensive pytest tests for report generation."""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiler import FluxProfiler, ProfileReport


# ── JSON report ────────────────────────────────────────

class TestJSONReport:
    def test_valid_json(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        j = report.to_json()
        # Should not raise
        data = json.loads(j)
        assert isinstance(data, dict)

    def test_json_contains_total_cycles(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        data = json.loads(report.to_json())
        assert "total_cycles" in data
        assert data["total_cycles"] == 2

    def test_json_contains_total_instructions(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        data = json.loads(report.to_json())
        assert "total_instructions" in data
        assert data["total_instructions"] == 2

    def test_json_contains_opcode_profiles(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        data = json.loads(report.to_json())
        assert "opcode_profiles" in data
        assert isinstance(data["opcode_profiles"], list)
        assert len(data["opcode_profiles"]) == 2  # MOVI + HALT

    def test_json_opcode_profile_fields(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        data = json.loads(report.to_json())
        op = data["opcode_profiles"][0]
        assert "opcode" in op
        assert "name" in op
        assert "count" in op
        assert "percentage" in op
        assert "cycles_estimate" in op

    def test_json_contains_hot_paths(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        data = json.loads(report.to_json())
        assert "hot_paths" in data

    def test_json_contains_register_usage(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        data = json.loads(report.to_json())
        assert "register_usage" in data
        assert isinstance(data["register_usage"], list)

    def test_json_contains_program_size(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        data = json.loads(report.to_json())
        assert data["program_size"] == 4

    def test_json_contains_ipc(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        data = json.loads(report.to_json())
        assert "ipc" in data
        assert data["ipc"] > 0

    def test_json_pretty_printed(self):
        """JSON should be indented for readability."""
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        j = report.to_json()
        assert "\n" in j  # indented JSON has newlines
        assert "  " in j   # has spaces for indentation


# ── Markdown report ────────────────────────────────────

class TestMarkdownReport:
    def test_contains_heading(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "# FLUX Profile Report" in md

    def test_contains_total_cycles(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "**Total Cycles:** 2" in md

    def test_contains_total_instructions(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "**Total Instructions:** 2" in md

    def test_contains_program_size(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "**Program Size:** 4 bytes" in md

    def test_contains_ipc(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "**IPC:**" in md

    def test_contains_opcode_table(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "Opcode Distribution" in md
        assert "| Opcode | Count |" in md

    def test_opcode_table_contains_movi(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "MOVI" in md

    def test_opcode_table_limited_to_10(self):
        """Table should show top 10 opcodes."""
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        # Count table rows (excluding header)
        table_rows = [line for line in md.split('\n') if line.startswith('|') and 'Opcode' not in line and '---' not in line]
        assert len(table_rows) <= 10

    def test_contains_hot_paths_section(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "Hot Paths" in md

    def test_hot_paths_limited_to_5(self):
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        md = report.to_markdown()
        hp_lines = [line for line in md.split('\n') if line.startswith('- **PC')]
        assert len(hp_lines) <= 5

    def test_contains_register_usage_section(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "Register Usage" in md

    def test_register_usage_limited_to_8(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        md = report.to_markdown()
        reg_lines = [line for line in md.split('\n') if line.startswith('- R')]
        assert len(reg_lines) <= 8

    def test_no_hot_paths_when_short(self):
        """Should not show Hot Paths section for short programs."""
        p = FluxProfiler([0x00])
        report = p.profile()
        md = report.to_markdown()
        assert "Hot Paths" not in md

    def test_markdown_returns_string(self):
        p = FluxProfiler([0x18, 0, 42, 0x00])
        report = p.profile()
        assert isinstance(report.to_markdown(), str)


# ── ProfileReport post_init ────────────────────────────

class TestProfileReportPostInit:
    def test_ipc_computed_in_post_init(self):
        report = ProfileReport(
            total_cycles=10,
            total_instructions=5,
            opcode_profiles=[],
            hot_paths=[],
            register_usage=[],
            program_size=10,
        )
        assert report.ipc == 0.5

    def test_ipc_zero_when_no_cycles(self):
        report = ProfileReport(
            total_cycles=0,
            total_instructions=0,
            opcode_profiles=[],
            hot_paths=[],
            register_usage=[],
            program_size=0,
        )
        assert report.ipc == 0.0

    def test_ipc_one_when_equal(self):
        report = ProfileReport(
            total_cycles=5,
            total_instructions=5,
            opcode_profiles=[],
            hot_paths=[],
            register_usage=[],
            program_size=10,
        )
        assert report.ipc == 1.0


# ── complex program scenarios ──────────────────────────

class TestComplexPrograms:
    def test_factorial_profile(self):
        """Profile 5! = 120 factorial loop."""
        p = FluxProfiler([0x18, 0, 5, 0x18, 1, 1, 0x22, 1, 1, 0, 0x09, 0, 0x3D, 0, 0xFA, 0, 0x00])
        report = p.profile()
        assert report.total_instructions > 10
        assert report.total_cycles > 10
        # Verify the profiler captured loop-heavy opcodes
        op_names = {o.name for o in report.opcode_profiles}
        assert "MUL" in op_names
        assert "DEC" in op_names

    def test_many_nops_profile(self):
        """Profile 100 NOPs + HALT."""
        bytecode = [0x01] * 100 + [0x00]
        p = FluxProfiler(bytecode)
        report = p.profile()
        assert report.total_instructions == 101
        assert report.total_cycles == 101  # all cost 1 cycle

    def test_nested_operations_profile(self):
        """Profile program with many different opcodes."""
        bytecode = [
            0x18, 0, 10,  # MOVI R0, 10
            0x18, 1, 20,  # MOVI R1, 20
            0x18, 2, 0,   # MOVI R2, 0
            0x20, 2, 0, 1,  # ADD R2, R0, R1
            0x21, 2, 2, 0,  # SUB R2, R2, R0
            0x22, 2, 2, 1,  # MUL R2, R2, R1
            0x00,            # HALT
        ]
        p = FluxProfiler(bytecode)
        report = p.profile()
        op_names = {o.name for o in report.opcode_profiles}
        assert "MOVI" in op_names
        assert "ADD" in op_names
        assert "SUB" in op_names
        assert "MUL" in op_names
        assert "HALT" in op_names

    def test_division_profile(self):
        """Profile program with DIV."""
        bytecode = [0x18, 0, 100, 0x18, 1, 4, 0x23, 2, 0, 1, 0x00]
        p = FluxProfiler(bytecode)
        report = p.profile()
        assert report.total_instructions == 4
        # DIV costs 4 cycles: MOVI(1) + MOVI(1) + DIV(4) + HALT(1) = 7
        assert report.total_cycles == 7
