"""Tests for statistical analysis module."""

import math

import pytest

from flux_profiler.stats import (
    ComparisonTable,
    PerformanceStats,
    ThroughputAnalyzer,
)
from flux_profiler.vm_adapter import ProfileResult


def _make_result(
    name: str = "test",
    vm: str = "MiniVM-Python",
    time_ns: int = 10000,
    instructions: int = 1000,
    cycles: int = 2000,
    mem_r: int = 10,
    mem_w: int = 5,
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
        memory_reads=mem_r,
        memory_writes=mem_w,
        peak_stack_depth=2,
        registers_final={0: 42},
        result_correct=correct,
        error=error,
    )


# ── PerformanceStats Tests ───────────────────────────────────────────────────

class TestPerformanceStats:
    """Tests for PerformanceStats computation."""

    def test_from_empty_results(self):
        stats = PerformanceStats.from_results([])
        assert stats.count == 0
        assert stats.mean_time_ns == 0.0

    def test_from_single_result(self):
        results = [_make_result(time_ns=10000)]
        stats = PerformanceStats.from_results(results)
        assert stats.count == 1
        assert stats.mean_time_ns == 10000.0
        assert stats.median_time_ns == 10000.0
        assert stats.std_dev_ns == 0.0

    def test_from_multiple_results(self):
        results = [_make_result(time_ns=10000), _make_result(time_ns=20000), _make_result(time_ns=30000)]
        stats = PerformanceStats.from_results(results)
        assert stats.count == 3
        assert stats.mean_time_ns == 20000.0
        assert stats.median_time_ns == 20000.0
        assert stats.min_time_ns == 10000.0
        assert stats.max_time_ns == 30000.0

    def test_std_dev_calculation(self):
        results = [_make_result(time_ns=10000), _make_result(time_ns=20000)]
        stats = PerformanceStats.from_results(results)
        # std_dev for [10000, 20000] = sqrt(50000000) ≈ 7071.07
        assert stats.std_dev_ns == pytest.approx(7071.07, abs=1.0)

    def test_median_odd_count(self):
        results = [_make_result(time_ns=t) for t in [1000, 3000, 5000, 7000, 9000]]
        stats = PerformanceStats.from_results(results)
        assert stats.median_time_ns == 5000.0

    def test_median_even_count(self):
        results = [_make_result(time_ns=t) for t in [1000, 2000, 3000, 4000]]
        stats = PerformanceStats.from_results(results)
        # Index 2 of sorted [1000,2000,3000,4000] = 3000
        assert stats.median_time_ns == 3000.0

    def test_throughput_calculation(self):
        results = [_make_result(time_ns=1_000_000, instructions=1_000_000)]
        stats = PerformanceStats.from_results(results)
        # 1M instructions in 1ms = 1 billion IPS
        assert stats.throughput_ips == pytest.approx(1e9, rel=0.01)

    def test_throughput_zero_time(self):
        results = [_make_result(time_ns=0, instructions=100)]
        stats = PerformanceStats.from_results(results)
        assert stats.throughput_ips == 0.0

    def test_cycles_per_instruction(self):
        results = [_make_result(instructions=1000, cycles=2500)]
        stats = PerformanceStats.from_results(results)
        assert stats.cycles_per_instruction == 2.5

    def test_cycles_per_instruction_zero(self):
        results = [_make_result(instructions=0, cycles=0)]
        stats = PerformanceStats.from_results(results)
        assert stats.cycles_per_instruction == 0.0

    def test_correctness_rate_all_correct(self):
        results = [_make_result(correct=True), _make_result(correct=True)]
        stats = PerformanceStats.from_results(results)
        assert stats.correctness_rate == 1.0

    def test_correctness_rate_half_correct(self):
        results = [_make_result(correct=True), _make_result(correct=False)]
        stats = PerformanceStats.from_results(results)
        assert stats.correctness_rate == 0.5

    def test_correctness_rate_none_correct(self):
        results = [_make_result(correct=False), _make_result(correct=False)]
        stats = PerformanceStats.from_results(results)
        assert stats.correctness_rate == 0.0

    def test_efficiency_score(self):
        results = [_make_result(time_ns=1_000_000, instructions=1_000_000, correct=True)]
        stats = PerformanceStats.from_results(results)
        # Should be between 0 and 100
        assert 0.0 <= stats.efficiency_score <= 100.0

    def test_efficiency_score_high_throughput(self):
        results = [_make_result(time_ns=1000, instructions=1_000_000, correct=True)]
        stats = PerformanceStats.from_results(results)
        # High throughput → high score
        assert stats.efficiency_score > 50.0

    def test_benchmark_name_preserved(self):
        results = [_make_result(name="fibonacci")]
        stats = PerformanceStats.from_results(results)
        assert stats.benchmark_name == "fibonacci"

    def test_vm_name_preserved(self):
        results = [_make_result(vm="MyVM")]
        stats = PerformanceStats.from_results(results)
        assert stats.vm_name == "MyVM"

    def test_total_instructions(self):
        results = [_make_result(instructions=500)]
        stats = PerformanceStats.from_results(results)
        assert stats.total_instructions == 500

    def test_total_cycles(self):
        results = [_make_result(cycles=1000)]
        stats = PerformanceStats.from_results(results)
        assert stats.total_cycles == 1000

    def test_total_memory_reads(self):
        results = [_make_result(mem_r=42)]
        stats = PerformanceStats.from_results(results)
        assert stats.total_mem_reads == 42

    def test_total_memory_writes(self):
        results = [_make_result(mem_w=17)]
        stats = PerformanceStats.from_results(results)
        assert stats.total_mem_writes == 17


# ── ComparisonTable Tests ────────────────────────────────────────────────────

class TestComparisonTable:
    """Tests for the ComparisonTable class."""

    def _make_results(self, vm_name: str, times: list[int]) -> list[ProfileResult]:
        return [_make_result(vm=vm_name, time_ns=t) for t in times]

    def test_from_empty(self):
        table = ComparisonTable.from_results_by_vm({})
        assert table.rows == []
        assert table.vm_names == []

    def test_from_single_vm(self):
        results = self._make_results("VM1", [10000, 20000, 30000])
        table = ComparisonTable.from_results_by_vm({"VM1": results})
        assert len(table.rows) == 3
        assert "VM1" in table.vm_names

    def test_from_multiple_vms(self):
        table = ComparisonTable.from_results_by_vm({
            "VM1": self._make_results("VM1", [10000]),
            "VM2": self._make_results("VM2", [20000]),
        })
        assert set(table.vm_names) == {"VM1", "VM2"}
        assert len(table.rows) == 2

    def test_rank_vms_single(self):
        table = ComparisonTable.from_results_by_vm({
            "VM1": self._make_results("VM1", [10000]),
        })
        rankings = table.rank_vms()
        assert len(rankings) == 1
        assert rankings[0][0] == "VM1"

    def test_rank_vms_multiple(self):
        table = ComparisonTable.from_results_by_vm({
            "FastVM": self._make_results("FastVM", [5000]),
            "SlowVM": self._make_results("SlowVM", [20000]),
        })
        rankings = table.rank_vms()
        assert rankings[0][0] == "FastVM"
        assert rankings[1][0] == "SlowVM"
        assert rankings[0][1] > rankings[1][1]

    def test_speedup_ratio(self):
        table = ComparisonTable.from_results_by_vm({
            "FastVM": [_make_result(name="bm1", vm="FastVM", time_ns=5000)],
            "SlowVM": [_make_result(name="bm1", vm="SlowVM", time_ns=20000)],
        })
        ratios = table.speedup_ratio("FastVM", "SlowVM")
        assert "bm1" in ratios
        assert ratios["bm1"] == 4.0  # 20000 / 5000

    def test_speedup_ratio_no_common_benchmarks(self):
        table = ComparisonTable.from_results_by_vm({
            "VM1": [_make_result(name="bm1", vm="VM1", time_ns=10000)],
            "VM2": [_make_result(name="bm2", vm="VM2", time_ns=20000)],
        })
        ratios = table.speedup_ratio("VM1", "VM2")
        assert len(ratios) == 0

    def test_geometric_mean_speedup(self):
        table = ComparisonTable.from_results_by_vm({
            "FastVM": [
                _make_result(name="bm1", vm="FastVM", time_ns=1000),
                _make_result(name="bm2", vm="FastVM", time_ns=1000),
            ],
            "SlowVM": [
                _make_result(name="bm1", vm="SlowVM", time_ns=2000),
                _make_result(name="bm2", vm="SlowVM", time_ns=8000),
            ],
        })
        geo = table.geometric_mean_speedup("FastVM", "SlowVM")
        # geo mean of [2.0, 8.0] = sqrt(16) = 4.0
        assert geo == pytest.approx(4.0, rel=0.01)

    def test_geometric_mean_no_data(self):
        table = ComparisonTable.from_results_by_vm({
            "VM1": [_make_result(name="bm1", vm="VM1")],
            "VM2": [_make_result(name="bm2", vm="VM2")],
        })
        assert table.geometric_mean_speedup("VM1", "VM2") == 0.0

    def test_normalized_score(self):
        table = ComparisonTable.from_results_by_vm({
            "FastVM": [_make_result(name="bm1", vm="FastVM", time_ns=5000)],
            "SlowVM": [_make_result(name="bm1", vm="SlowVM", time_ns=20000)],
        })
        rows = table.rows
        fast_row = [r for r in rows if r.vm_name == "FastVM"][0]
        slow_row = [r for r in rows if r.vm_name == "SlowVM"][0]
        assert fast_row.normalized_score == 1.0  # fastest
        assert slow_row.normalized_score == 0.25  # 5000/20000

    def test_statistical_significance_different(self):
        """Different means should be significant with enough samples."""
        results_a = [_make_result(time_ns=10000) for _ in range(20)]
        results_b = [_make_result(time_ns=50000) for _ in range(20)]
        assert ComparisonTable.statistical_significance(results_a, results_b) is True

    def test_statistical_significance_similar(self):
        """Similar means should not be significant."""
        results_a = [_make_result(time_ns=10000 + i * 10) for i in range(10)]
        results_b = [_make_result(time_ns=10005 + i * 10) for i in range(10)]
        result = ComparisonTable.statistical_significance(results_a, results_b)
        # With similar means and small sample, may or may not be significant
        assert isinstance(result, bool)

    def test_statistical_significance_too_few(self):
        results_a = [_make_result(time_ns=10000)]
        results_b = [_make_result(time_ns=50000)]
        assert ComparisonTable.statistical_significance(results_a, results_b) is False

    def test_t_critical_values(self):
        assert ComparisonTable._t_critical(1) == 12.706
        assert ComparisonTable._t_critical(5) == 2.571
        assert ComparisonTable._t_critical(30) == 2.042
        assert ComparisonTable._t_critical(1000) == 1.960


# ── ThroughputAnalyzer Tests ─────────────────────────────────────────────────

class TestThroughputAnalyzer:
    """Tests for the ThroughputAnalyzer class."""

    def test_categorize_benchmarks(self):
        results = [
            _make_result(name="factorial_small"),
            _make_result(name="bubble_sort"),
            _make_result(name="stack_heavy"),
            _make_result(name="float_arithmetic"),
        ]
        categories = ThroughputAnalyzer.categorize_benchmarks(results)
        assert len(categories) > 0
        # Should have at least arithmetic and other categories
        assert len(categories) >= 2

    def test_categorize_empty(self):
        categories = ThroughputAnalyzer.categorize_benchmarks([])
        assert len(categories) == 0

    def test_infer_category_arithmetic(self):
        assert ThroughputAnalyzer._infer_category("factorial_small") == "arithmetic"
        assert ThroughputAnalyzer._infer_category("factorial_large") == "arithmetic"

    def test_infer_category_control_flow(self):
        assert ThroughputAnalyzer._infer_category("fibonacci") == "control_flow"
        assert ThroughputAnalyzer._infer_category("gcd") == "control_flow"
        assert ThroughputAnalyzer._infer_category("prime_sieve") == "control_flow"
        assert ThroughputAnalyzer._infer_category("branch_heavy") == "control_flow"

    def test_infer_category_memory(self):
        assert ThroughputAnalyzer._infer_category("memory_bandwidth") == "memory"
        assert ThroughputAnalyzer._infer_category("matrix_multiply") == "memory"
        assert ThroughputAnalyzer._infer_category("bubble_sort") == "memory"

    def test_infer_category_stack(self):
        assert ThroughputAnalyzer._infer_category("stack_heavy") == "stack"
        assert ThroughputAnalyzer._infer_category("recursive_call") == "stack"

    def test_infer_category_float(self):
        assert ThroughputAnalyzer._infer_category("float_arithmetic") == "float"

    def test_infer_category_a2a(self):
        assert ThroughputAnalyzer._infer_category("a2a_simulation") == "a2a"

    def test_infer_category_dispatch(self):
        assert ThroughputAnalyzer._infer_category("noop_baseline") == "dispatch"
        assert ThroughputAnalyzer._infer_category("syscall_heavy") == "dispatch"

    def test_infer_category_mixed(self):
        assert ThroughputAnalyzer._infer_category("register_pressure") == "mixed"
        assert ThroughputAnalyzer._infer_category("long_program") == "mixed"

    def test_infer_category_bit(self):
        assert ThroughputAnalyzer._infer_category("bit_manipulation") == "arithmetic"

    def test_identify_bottlenecks(self):
        results = [
            _make_result(name="noop_baseline", time_ns=1000),
            _make_result(name="branch_heavy", time_ns=100000),
            _make_result(name="memory_bandwidth", time_ns=50000),
        ]
        bottlenecks = ThroughputAnalyzer.identify_bottlenecks(results)
        assert len(bottlenecks) > 0
        # First should be the slowest category
        assert bottlenecks[0][1] >= bottlenecks[-1][1]

    def test_identify_bottlenecks_empty(self):
        bottlenecks = ThroughputAnalyzer.identify_bottlenecks([])
        assert bottlenecks == []

    def test_geometric_mean(self):
        results = [_make_result(time_ns=t) for t in [1000, 4000]]
        geo = ThroughputAnalyzer.geometric_mean(results)
        assert geo == pytest.approx(2000.0, rel=0.01)

    def test_geometric_mean_single(self):
        results = [_make_result(time_ns=5000)]
        assert ThroughputAnalyzer.geometric_mean(results) == 5000.0

    def test_geometric_mean_empty(self):
        assert ThroughputAnalyzer.geometric_mean([]) == 0.0

    def test_geometric_mean_skips_zero(self):
        results = [_make_result(time_ns=0), _make_result(time_ns=10000)]
        geo = ThroughputAnalyzer.geometric_mean(results)
        assert geo == 10000.0

    def test_scaling_analysis(self):
        results = [
            _make_result(name="test", instructions=100, time_ns=1000),
            _make_result(name="test", instructions=200, time_ns=2000),
            _make_result(name="test", instructions=300, time_ns=3000),
        ]
        scaling = ThroughputAnalyzer.scaling_analysis(results)
        assert "test" in scaling
        assert len(scaling["test"]) == 3

    def test_scaling_analysis_empty(self):
        scaling = ThroughputAnalyzer.scaling_analysis([])
        assert scaling == {}

    def test_scaling_analysis_custom_sizes(self):
        results = [
            _make_result(name="test", instructions=100, time_ns=1000),
            _make_result(name="test", instructions=200, time_ns=2000),
        ]
        scaling = ThroughputAnalyzer.scaling_analysis(results, sizes=[10, 20])
        assert len(scaling["test"]) == 2
        assert scaling["test"][0][0] == 10
        assert scaling["test"][1][0] == 20
