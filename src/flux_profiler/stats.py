"""Statistical analysis for FLUX profiler results.

Provides performance statistics, comparison tables, throughput analysis,
and significance testing for benchmark results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PerformanceStats:
    """Statistical summary of benchmark execution results."""

    benchmark_name: str
    vm_name: str
    count: int = 0
    mean_time_ns: float = 0.0
    median_time_ns: float = 0.0
    min_time_ns: float = 0.0
    max_time_ns: float = 0.0
    std_dev_ns: float = 0.0
    total_instructions: int = 0
    total_cycles: int = 0
    total_mem_reads: int = 0
    total_mem_writes: int = 0
    throughput_ips: float = 0.0  # instructions per second
    cycles_per_instruction: float = 0.0
    efficiency_score: float = 0.0
    correctness_rate: float = 0.0

    @classmethod
    def from_results(cls, results: list) -> PerformanceStats:
        """Compute statistics from a list of ProfileResult objects.

        Args:
            results: List of ProfileResult instances from repeated runs.

        Returns:
            PerformanceStats with computed metrics.
        """
        if not results:
            return cls()

        # Import here to avoid circular dependency
        from flux_profiler.vm_adapter import ProfileResult

        times = [r.wall_time_ns for r in results]
        n = len(times)
        name = results[0].benchmark_name if results else ""
        vm = results[0].vm_name if results else ""

        # Central tendency
        mean_t = sum(times) / n
        sorted_times = sorted(times)
        median_t = sorted_times[n // 2]
        min_t = sorted_times[0]
        max_t = sorted_times[-1]

        # Standard deviation
        if n > 1:
            variance = sum((t - mean_t) ** 2 for t in times) / (n - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0

        # Aggregated counters (use median result's counters)
        median_result = sorted_times  # already sorted
        median_idx = n // 2
        median_r = results[median_idx]

        total_instr = median_r.instructions_executed
        total_cycles = median_r.cycles_used
        total_reads = median_r.memory_reads
        total_writes = median_r.memory_writes

        # Throughput
        throughput = total_instr / (mean_t / 1e9) if mean_t > 0 else 0.0

        # Cycles per instruction
        cpi = total_cycles / total_instr if total_instr > 0 else 0.0

        # Efficiency score: normalized metric (0-100)
        # Combines speed (inversely related to time) and correctness
        correct_count = sum(1 for r in results if r.result_correct)
        correctness = correct_count / n if n > 0 else 0.0

        # Efficiency: higher throughput + correctness = higher score
        # Normalize throughput to a 0-100 scale relative to a baseline
        # Assume 1 billion IPS as baseline
        efficiency = min(100.0, (throughput / 1e9) * 50 + correctness * 50)

        return cls(
            benchmark_name=name,
            vm_name=vm,
            count=n,
            mean_time_ns=mean_t,
            median_time_ns=median_t,
            min_time_ns=min_t,
            max_time_ns=max_t,
            std_dev_ns=std_dev,
            total_instructions=total_instr,
            total_cycles=total_cycles,
            total_mem_reads=total_reads,
            total_mem_writes=total_writes,
            throughput_ips=throughput,
            cycles_per_instruction=cpi,
            efficiency_score=efficiency,
            correctness_rate=correctness,
        )


@dataclass
class ComparisonRow:
    """A row in a comparison table for a single benchmark."""
    benchmark_name: str
    vm_name: str
    median_time_ns: float
    mean_time_ns: float
    std_dev_ns: float
    throughput_ips: float
    instructions: int
    cycles: int
    correctness: bool
    normalized_score: float = 0.0


class ComparisonTable:
    """Build and analyze comparison tables across VM implementations."""

    def __init__(self) -> None:
        self._rows: list[ComparisonRow] = []
        self._vm_names: set[str] = set()

    @classmethod
    def from_results_by_vm(
        cls, results_by_vm: dict[str, list]
    ) -> ComparisonTable:
        """Build a comparison table from per-VM result lists.

        Args:
            results_by_vm: Dict mapping VM name to list of ProfileResult.

        Returns:
            Populated ComparisonTable.
        """
        table = cls()
        # Collect all stats
        all_stats: list[PerformanceStats] = []
        for vm_name, results in results_by_vm.items():
            table._vm_names.add(vm_name)
            for result in results:
                stats = PerformanceStats.from_results([result])
                all_stats.append(stats)

        if not all_stats:
            return table

        # Find time range for normalization
        times = [s.median_time_ns for s in all_stats if s.median_time_ns > 0]
        min_time = min(times) if times else 1.0

        # Build rows
        for stats in all_stats:
            # Find the result for correctness
            from flux_profiler.vm_adapter import ProfileResult
            matching = None
            for vm_name, results in results_by_vm.items():
                if vm_name == stats.vm_name:
                    for r in results:
                        if r.benchmark_name == stats.benchmark_name:
                            matching = r
                            break
                if matching:
                    break

            row = ComparisonRow(
                benchmark_name=stats.benchmark_name,
                vm_name=stats.vm_name,
                median_time_ns=stats.median_time_ns,
                mean_time_ns=stats.mean_time_ns,
                std_dev_ns=stats.std_dev_ns,
                throughput_ips=stats.throughput_ips,
                instructions=stats.total_instructions,
                cycles=stats.total_cycles,
                correctness=matching.result_correct if matching else False,
                normalized_score=min_time / stats.median_time_ns if stats.median_time_ns > 0 else 0.0,
            )
            table._rows.append(row)

        return table

    @property
    def rows(self) -> list[ComparisonRow]:
        return list(self._rows)

    @property
    def vm_names(self) -> list[str]:
        return sorted(self._vm_names)

    def rank_vms(self) -> list[tuple[str, float]]:
        """Rank VMs by average normalized score across benchmarks.

        Returns:
            List of (vm_name, avg_score) tuples, sorted best to worst.
        """
        vm_scores: dict[str, list[float]] = {}
        for row in self._rows:
            vm_scores.setdefault(row.vm_name, []).append(row.normalized_score)

        rankings = []
        for vm_name, scores in vm_scores.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            rankings.append((vm_name, avg))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def speedup_ratio(self, vm_a: str, vm_b: str) -> dict[str, float]:
        """Compute speedup ratios: how much faster is vm_a than vm_b?

        Returns:
            Dict mapping benchmark_name to speedup ratio (>1 means vm_a is faster).
        """
        ratios: dict[str, float] = {}
        a_times: dict[str, float] = {}
        b_times: dict[str, float] = {}

        for row in self._rows:
            if row.vm_name == vm_a:
                a_times[row.benchmark_name] = row.median_time_ns
            elif row.vm_name == vm_b:
                b_times[row.benchmark_name] = row.median_time_ns

        for bm_name in a_times:
            if bm_name in b_times and b_times[bm_name] > 0:
                ratios[bm_name] = b_times[bm_name] / a_times[bm_name]

        return ratios

    def geometric_mean_speedup(self, vm_a: str, vm_b: str) -> float:
        """Compute geometric mean of speedup ratios across benchmarks."""
        ratios = self.speedup_ratio(vm_a, vm_b)
        if not ratios:
            return 0.0
        log_sum = sum(math.log(r) for r in ratios.values() if r > 0)
        return math.exp(log_sum / len(ratios))

    @staticmethod
    def statistical_significance(results_a: list, results_b: list, alpha: float = 0.05) -> bool:
        """Simple two-sample t-test for statistical significance.

        Tests whether the means of two sets of results are significantly different.

        Args:
            results_a: First set of ProfileResult (wall_time_ns values).
            results_b: Second set of ProfileResult (wall_time_ns values).
            alpha: Significance level.

        Returns:
            True if the difference is statistically significant.
        """
        times_a = [r.wall_time_ns for r in results_a]
        times_b = [r.wall_time_ns for r in results_b]

        n_a = len(times_a)
        n_b = len(times_b)

        if n_a < 2 or n_b < 2:
            return False

        mean_a = sum(times_a) / n_a
        mean_b = sum(times_b) / n_b

        var_a = sum((t - mean_a) ** 2 for t in times_a) / (n_a - 1)
        var_b = sum((t - mean_b) ** 2 for t in times_b) / (n_b - 1)

        # Pooled standard error
        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return False

        # t-statistic
        t_stat = abs(mean_a - mean_b) / se

        # Degrees of freedom (Welch's approximation)
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        if denom == 0:
            return False
        df = num / denom

        # Approximate critical t-value for alpha=0.05 (two-tailed)
        # For large df, t_crit ≈ 1.96
        # For small df, use a lookup
        t_crit = ComparisonTable._t_critical(df, alpha)

        return t_stat > t_crit

    @staticmethod
    def _t_critical(df: float, alpha: float = 0.05) -> float:
        """Approximate t-critical value using a simplified lookup."""
        # Common t-critical values for alpha=0.05, two-tailed
        t_table = [
            (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776),
            (5, 2.571), (6, 2.447), (7, 2.365), (8, 2.306),
            (9, 2.262), (10, 2.228), (15, 2.131), (20, 2.086),
            (30, 2.042), (60, 2.000), (120, 1.980), (float('inf'), 1.960),
        ]
        # Find the closest entry
        for i, (t_df, t_val) in enumerate(t_table):
            if df <= t_df:
                return t_val
        return 1.960  # normal approximation


class ThroughputAnalyzer:
    """Analyze throughput patterns and identify performance bottlenecks."""

    @staticmethod
    def categorize_benchmarks(results: list) -> dict[str, list]:
        """Group results by benchmark category.

        Args:
            results: List of ProfileResult objects.

        Returns:
            Dict mapping category name to list of results.
        """
        from flux_profiler.benchmarks import BenchmarkWorkload
        from flux_profiler.vm_adapter import ProfileResult

        categories: dict[str, list] = {}
        for r in results:
            # Infer category from benchmark name
            cat = ThroughputAnalyzer._infer_category(r.benchmark_name)
            categories.setdefault(cat, []).append(r)
        return categories

    @staticmethod
    def _infer_category(name: str) -> str:
        """Infer benchmark category from its name."""
        name_lower = name.lower()
        if 'float' in name_lower:
            return 'float'
        if 'a2a' in name_lower or 'tell' in name_lower or 'bcast' in name_lower:
            return 'a2a'
        if 'stack' in name_lower or 'recursive' in name_lower:
            return 'stack'
        if 'memory' in name_lower or 'matrix' in name_lower or 'bubble' in name_lower:
            return 'memory'
        if 'branch' in name_lower or 'fibonacci' in name_lower or 'gcd' in name_lower or 'prime' in name_lower:
            return 'control_flow'
        if 'noop' in name_lower or 'syscall' in name_lower:
            return 'dispatch'
        if 'bit' in name_lower:
            return 'arithmetic'
        if 'factorial' in name_lower or 'sort' in name_lower:
            return 'arithmetic'
        return 'mixed'

    @staticmethod
    def identify_bottlenecks(results: list) -> list[tuple[str, float]]:
        """Identify which benchmark categories are the slowest.

        Returns:
            List of (category, avg_time_ns) sorted slowest to fastest.
        """
        categories = ThroughputAnalyzer.categorize_benchmarks(results)
        cat_times: list[tuple[str, float]] = []
        for cat, cat_results in categories.items():
            times = [r.wall_time_ns for r in cat_results if r.error is None]
            if times:
                avg = sum(times) / len(times)
                cat_times.append((cat, avg))
        cat_times.sort(key=lambda x: x[1], reverse=True)
        return cat_times

    @staticmethod
    def scaling_analysis(results: list, sizes: Optional[list[int]] = None) -> dict[str, list[tuple[int, float]]]:
        """Analyze how performance scales with input size.

        Args:
            results: List of ProfileResult objects.
            sizes: Input sizes to group by (default: [1, 5, 10, 20, 50, 100]).

        Returns:
            Dict mapping benchmark name to [(size, time_ns), ...].
        """
        if sizes is None:
            sizes = [1, 5, 10, 20, 50, 100]

        # Group results by benchmark name
        by_name: dict[str, list] = {}
        for r in results:
            by_name.setdefault(r.benchmark_name, []).append(r)

        scaling: dict[str, list[tuple[int, float]]] = {}
        for name, name_results in by_name.items():
            # Use instructions_executed as a proxy for "size"
            sorted_results = sorted(name_results, key=lambda r: r.instructions_executed)
            n = min(len(sizes), len(sorted_results))
            scaling[name] = [(sizes[i], sorted_results[i].wall_time_ns) for i in range(n)]

        return scaling

    @staticmethod
    def geometric_mean(results: list) -> float:
        """Compute geometric mean of wall_time_ns across results.

        Returns:
            Geometric mean time in nanoseconds.
        """
        times = [r.wall_time_ns for r in results if r.wall_time_ns > 0]
        if not times:
            return 0.0
        log_sum = sum(math.log(t) for t in times)
        return math.exp(log_sum / len(times))
