"""Tests for the benchmark report generation module."""

import csv
import io
import json

import pytest

from flux_profiler.profiler import Profiler
from flux_profiler.reporter import BenchmarkReporter
from flux_profiler.vm_adapter import MiniVMAdapter, ProfileResult


def _make_result(
    name: str = "test",
    vm: str = "MiniVM-Python",
    time_ns: int = 10000,
    instructions: int = 1000,
    cycles: int = 2000,
    correct: bool = True,
    error: str | None = None,
) -> ProfileResult:
    """Helper to create a ProfileResult for testing."""
    return ProfileResult(
        benchmark_name=name,
        vm_name=vm,
        wall_time_ns=time_ns,
        instructions_executed=instructions,
        cycles_used=cycles,
        memory_reads=10,
        memory_writes=5,
        peak_stack_depth=2,
        registers_final={0: 42},
        result_correct=correct,
        error=error,
    )


class TestBenchmarkReporterSummary:
    """Tests for the summary report method."""

    def test_summary_empty(self):
        summary = BenchmarkReporter.summary([])
        assert summary["total_benchmarks"] == 0
        assert summary["total_vms"] == 0
        assert summary["total_runs"] == 0

    def test_summary_single_result(self):
        results = [_make_result()]
        summary = BenchmarkReporter.summary(results)
        assert summary["total_benchmarks"] == 1
        assert summary["total_vms"] == 1
        assert summary["total_runs"] == 1
        assert summary["successful_runs"] == 1
        assert summary["correct_runs"] == 1

    def test_summary_multiple_vms(self):
        results = [
            _make_result(vm="VM1"),
            _make_result(vm="VM2"),
        ]
        summary = BenchmarkReporter.summary(results)
        assert summary["total_vms"] == 2

    def test_summary_failed_runs(self):
        results = [_make_result(error="crash")]
        summary = BenchmarkReporter.summary(results)
        assert summary["failed_runs"] == 1
        assert summary["successful_runs"] == 0

    def test_summary_incorrect_runs(self):
        results = [_make_result(correct=False)]
        summary = BenchmarkReporter.summary(results)
        assert summary["correct_runs"] == 0
        assert summary["successful_runs"] == 1

    def test_summary_total_time(self):
        results = [_make_result(time_ns=5000), _make_result(time_ns=3000)]
        summary = BenchmarkReporter.summary(results)
        assert summary["total_time_ns"] == 8000
        assert summary["total_time_us"] == 8.0
        assert summary["total_time_ms"] == 0.008

    def test_summary_total_instructions(self):
        results = [_make_result(instructions=100), _make_result(instructions=200)]
        summary = BenchmarkReporter.summary(results)
        assert summary["total_instructions"] == 300

    def test_summary_throughput(self):
        results = [_make_result(time_ns=1_000_000, instructions=1_000_000)]
        summary = BenchmarkReporter.summary(results)
        assert summary["avg_throughput_ips"] > 0

    def test_summary_failure_details(self):
        results = [_make_result(name="bm1", vm="VM1", error="timeout")]
        summary = BenchmarkReporter.summary(results)
        assert len(summary["failure_details"]) == 1
        assert summary["failure_details"][0]["benchmark"] == "bm1"
        assert summary["failure_details"][0]["error"] == "timeout"

    def test_summary_vm_names_sorted(self):
        results = [_make_result(vm="Charlie"), _make_result(vm="Alpha"), _make_result(vm="Bravo")]
        summary = BenchmarkReporter.summary(results)
        assert summary["vm_names"] == ["Alpha", "Bravo", "Charlie"]

    def test_summary_benchmark_names_sorted(self):
        results = [_make_result(name="zebra"), _make_result(name="alpha"), _make_result(name="middle")]
        summary = BenchmarkReporter.summary(results)
        assert summary["benchmark_names"] == ["alpha", "middle", "zebra"]


class TestBenchmarkReporterMarkdown:
    """Tests for Markdown report generation."""

    def test_markdown_contains_header(self):
        results = [_make_result()]
        md = BenchmarkReporter.to_markdown(results)
        assert "# FLUX VM Benchmark Report" in md

    def test_markdown_contains_summary_section(self):
        results = [_make_result()]
        md = BenchmarkReporter.to_markdown(results)
        assert "## Summary" in md

    def test_markdown_contains_results_section(self):
        results = [_make_result()]
        md = BenchmarkReporter.to_markdown(results)
        assert "## Results" in md

    def test_markdown_contains_table(self):
        results = [_make_result()]
        md = BenchmarkReporter.to_markdown(results)
        assert "| Benchmark |" in md
        assert "|-----------|" in md

    def test_markdown_contains_benchmark_name(self):
        results = [_make_result(name="factorial_small")]
        md = BenchmarkReporter.to_markdown(results)
        assert "factorial_small" in md

    def test_markdown_correctness_marker(self):
        results = [_make_result(correct=True)]
        md = BenchmarkReporter.to_markdown(results)
        assert "✓" in md

    def test_markdown_incorrectness_marker(self):
        results = [_make_result(correct=False)]
        md = BenchmarkReporter.to_markdown(results)
        assert "✗" in md

    def test_markdown_error_display(self):
        results = [_make_result(error="timeout")]
        md = BenchmarkReporter.to_markdown(results)
        assert "ERROR: timeout" in md

    def test_markdown_empty_results(self):
        md = BenchmarkReporter.to_markdown([])
        assert "# FLUX VM Benchmark Report" in md

    def test_markdown_bottleneck_section(self):
        results = [
            _make_result(name="noop_baseline", time_ns=1000),
            _make_result(name="branch_heavy", time_ns=100000),
        ]
        md = BenchmarkReporter.to_markdown(results)
        assert "## Bottleneck Analysis" in md

    def test_markdown_bottleneck_table(self):
        results = [
            _make_result(name="noop_baseline", time_ns=1000),
            _make_result(name="branch_heavy", time_ns=100000),
        ]
        md = BenchmarkReporter.to_markdown(results)
        assert "| Category |" in md


class TestBenchmarkReporterJSON:
    """Tests for JSON report generation."""

    def test_json_valid(self):
        results = [_make_result()]
        json_str = BenchmarkReporter.to_json(results)
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_json_fields(self):
        results = [_make_result(name="bm1", vm="VM1", time_ns=5000)]
        json_str = BenchmarkReporter.to_json(results)
        data = json.loads(json_str)
        entry = data[0]
        assert entry["benchmark_name"] == "bm1"
        assert entry["vm_name"] == "VM1"
        assert entry["wall_time_ns"] == 5000
        assert "instructions_executed" in entry
        assert "cycles_used" in entry
        assert "memory_reads" in entry
        assert "memory_writes" in entry
        assert "peak_stack_depth" in entry
        assert "result_correct" in entry
        assert "registers_final" in entry
        assert "error" in entry

    def test_json_multiple_results(self):
        results = [_make_result(name=f"bm{i}") for i in range(5)]
        json_str = BenchmarkReporter.to_json(results)
        data = json.loads(json_str)
        assert len(data) == 5

    def test_json_empty(self):
        json_str = BenchmarkReporter.to_json([])
        data = json.loads(json_str)
        assert data == []

    def test_json_error_field(self):
        results = [_make_result(error="crash")]
        json_str = BenchmarkReporter.to_json(results)
        data = json.loads(json_str)
        assert data[0]["error"] == "crash"

    def test_json_indentation(self):
        results = [_make_result()]
        json_str = BenchmarkReporter.to_json(results)
        assert "\n" in json_str  # pretty-printed


class TestBenchmarkReporterCSV:
    """Tests for CSV report generation."""

    def test_csv_valid(self):
        results = [_make_result()]
        csv_str = BenchmarkReporter.to_csv(results)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 2  # header + data
        assert rows[0][0] == "benchmark_name"

    def test_csv_fields(self):
        results = [_make_result(name="bm1", vm="VM1", time_ns=5000)]
        csv_str = BenchmarkReporter.to_csv(results)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        data = rows[1]
        assert data[0] == "bm1"
        assert data[1] == "VM1"
        assert data[2] == "5000"

    def test_csv_multiple_results(self):
        results = [_make_result(name=f"bm{i}") for i in range(5)]
        csv_str = BenchmarkReporter.to_csv(results)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 6  # header + 5 rows

    def test_csv_empty(self):
        csv_str = BenchmarkReporter.to_csv([])
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1  # header only

    def test_csv_r0_final(self):
        results = [_make_result()]
        csv_str = BenchmarkReporter.to_csv(results)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        # r0_final should be in the row
        assert "42" in rows[1]  # R0 = 42


class TestBenchmarkReporterComparison:
    """Tests for cross-VM comparison report."""

    def test_comparison_report_header(self):
        results_by_vm = {"VM1": [_make_result(vm="VM1")]}
        report = BenchmarkReporter.comparison_report(results_by_vm)
        assert "# FLUX VM Cross-VM Comparison Report" in report

    def test_comparison_report_rankings(self):
        results_by_vm = {
            "VM1": [_make_result(vm="VM1", time_ns=5000)],
            "VM2": [_make_result(vm="VM2", time_ns=10000)],
        }
        report = BenchmarkReporter.comparison_report(results_by_vm)
        assert "## VM Rankings" in report
        assert "| Rank |" in report

    def test_comparison_report_speedup(self):
        results_by_vm = {
            "FastVM": [_make_result(name="bm1", vm="FastVM", time_ns=5000)],
            "SlowVM": [_make_result(name="bm1", vm="SlowVM", time_ns=20000)],
        }
        report = BenchmarkReporter.comparison_report(results_by_vm)
        assert "## Speedup Analysis" in report
        assert "4.00x" in report

    def test_comparison_report_detailed(self):
        results_by_vm = {
            "VM1": [_make_result(name="bm1", vm="VM1")],
            "VM2": [_make_result(name="bm1", vm="VM2")],
        }
        report = BenchmarkReporter.comparison_report(results_by_vm)
        assert "## Detailed Results" in report
        assert "bm1" in report

    def test_comparison_report_single_vm(self):
        results_by_vm = {"VM1": [_make_result(vm="VM1")]}
        report = BenchmarkReporter.comparison_report(results_by_vm)
        assert "## VM Rankings" in report

    def test_comparison_report_empty(self):
        report = BenchmarkReporter.comparison_report({})
        assert "# FLUX VM Cross-VM Comparison Report" in report

    def test_comparison_report_correctness_markers(self):
        results_by_vm = {
            "VM1": [_make_result(name="bm1", vm="VM1", correct=True)],
            "VM2": [_make_result(name="bm1", vm="VM2", correct=False)],
        }
        report = BenchmarkReporter.comparison_report(results_by_vm)
        assert "✓" in report
        assert "✗" in report
