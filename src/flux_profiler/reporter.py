"""Report generation for FLUX profiler results.

Generates benchmark summaries in Markdown, JSON, CSV, and
cross-VM comparison formats.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Optional

from flux_profiler.stats import ComparisonTable, PerformanceStats, ThroughputAnalyzer
from flux_profiler.vm_adapter import ProfileResult


class BenchmarkReporter:
    """Generate reports from profiling results."""

    @staticmethod
    def summary(results: list[ProfileResult]) -> dict:
        """Generate a summary dict from a list of results.

        Returns:
            Dict with overview statistics.
        """
        if not results:
            return {"total_benchmarks": 0, "total_vms": 0}

        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        correct = [r for r in successful if r.result_correct]

        total_time = sum(r.wall_time_ns for r in successful)
        total_instructions = sum(r.instructions_executed for r in successful)
        total_cycles = sum(r.cycles_used for r in successful)

        vm_names = set(r.vm_name for r in results)
        bm_names = set(r.benchmark_name for r in results)

        return {
            "total_benchmarks": len(bm_names),
            "total_vms": len(vm_names),
            "total_runs": len(results),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "correct_runs": len(correct),
            "total_time_ns": total_time,
            "total_time_us": total_time / 1000,
            "total_time_ms": total_time / 1_000_000,
            "total_instructions": total_instructions,
            "total_cycles": total_cycles,
            "avg_throughput_ips": total_instructions / (total_time / 1e9) if total_time > 0 else 0,
            "vm_names": sorted(vm_names),
            "benchmark_names": sorted(bm_names),
            "failure_details": [
                {"benchmark": r.benchmark_name, "vm": r.vm_name, "error": r.error}
                for r in failed
            ],
        }

    @staticmethod
    def to_markdown(results: list[ProfileResult]) -> str:
        """Generate a Markdown report from profiling results.

        Includes a summary table and per-benchmark details.
        """
        lines: list[str] = []
        lines.append("# FLUX VM Benchmark Report")
        lines.append("")

        # Summary
        summ = BenchmarkReporter.summary(results)
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Benchmarks | {summ['total_benchmarks']} |")
        lines.append(f"| VMs | {summ['total_vms']} |")
        lines.append(f"| Total Runs | {summ['total_runs']} |")
        lines.append(f"| Successful | {summ['successful_runs']} |")
        lines.append(f"| Failed | {summ['failed_runs']} |")
        lines.append(f"| Correct | {summ['correct_runs']} |")
        lines.append(f"| Total Time | {summ['total_time_us']:.1f} μs |")
        lines.append(f"| Total Instructions | {summ['total_instructions']:,} |")
        lines.append(f"| Throughput | {summ['avg_throughput_ips']:,.0f} IPS |")
        lines.append("")

        # Results table
        lines.append("## Results")
        lines.append("")
        lines.append("| Benchmark | VM | Time (ns) | Instructions | Cycles | Memory R/W | Correct |")
        lines.append("|-----------|-----|-----------|--------------|--------|------------|---------|")
        for r in results:
            correct_str = "✓" if r.result_correct else "✗"
            error_str = f" (ERROR: {r.error})" if r.error else ""
            lines.append(
                f"| {r.benchmark_name} | {r.vm_name} | {r.wall_time_ns:,} | "
                f"{r.instructions_executed:,} | {r.cycles_used:,} | "
                f"{r.memory_reads}/{r.memory_writes} | {correct_str}{error_str} |"
            )
        lines.append("")

        # Bottleneck analysis
        if results:
            bottlenecks = ThroughputAnalyzer.identify_bottlenecks(results)
            if bottlenecks:
                lines.append("## Bottleneck Analysis")
                lines.append("")
                lines.append("| Category | Avg Time (ns) |")
                lines.append("|----------|---------------|")
                for cat, avg_time in bottlenecks:
                    lines.append(f"| {cat} | {avg_time:,.1f} |")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_json(results: list[ProfileResult]) -> str:
        """Generate a JSON report from profiling results.

        Each result is serialized with all its fields.
        """
        data = []
        for r in results:
            entry = {
                "benchmark_name": r.benchmark_name,
                "vm_name": r.vm_name,
                "wall_time_ns": r.wall_time_ns,
                "instructions_executed": r.instructions_executed,
                "cycles_used": r.cycles_used,
                "memory_reads": r.memory_reads,
                "memory_writes": r.memory_writes,
                "peak_stack_depth": r.peak_stack_depth,
                "registers_final": r.registers_final,
                "result_correct": r.result_correct,
                "error": r.error,
            }
            data.append(entry)
        return json.dumps(data, indent=2)

    @staticmethod
    def to_csv(results: list[ProfileResult]) -> str:
        """Generate a CSV report from profiling results.

        Returns:
            CSV-formatted string with headers and one row per result.
        """
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "benchmark_name", "vm_name", "wall_time_ns", "instructions_executed",
            "cycles_used", "memory_reads", "memory_writes", "peak_stack_depth",
            "result_correct", "error", "r0_final",
        ])
        for r in results:
            writer.writerow([
                r.benchmark_name,
                r.vm_name,
                r.wall_time_ns,
                r.instructions_executed,
                r.cycles_used,
                r.memory_reads,
                r.memory_writes,
                r.peak_stack_depth,
                r.result_correct,
                r.error or "",
                r.registers_final.get(0, ""),
            ])
        return output.getvalue()

    @staticmethod
    def comparison_report(results_by_vm: dict[str, list[ProfileResult]]) -> str:
        """Generate a cross-VM comparison report in Markdown.

        Args:
            results_by_vm: Dict mapping VM name to list of ProfileResult.

        Returns:
            Markdown string with comparison tables and rankings.
        """
        lines: list[str] = []
        lines.append("# FLUX VM Cross-VM Comparison Report")
        lines.append("")

        # Build comparison table
        table = ComparisonTable.from_results_by_vm(results_by_vm)
        vm_names = table.vm_names

        lines.append("## VM Rankings")
        lines.append("")
        rankings = table.rank_vms()
        lines.append("| Rank | VM | Avg Normalized Score |")
        lines.append("|------|-----|---------------------|")
        for rank, (vm_name, score) in enumerate(rankings, 1):
            lines.append(f"| {rank} | {vm_name} | {score:.4f} |")
        lines.append("")

        # Speedup ratios
        if len(vm_names) >= 2:
            lines.append("## Speedup Analysis")
            lines.append("")
            vm_a, vm_b = vm_names[0], vm_names[1]
            geo_mean = table.geometric_mean_speedup(vm_a, vm_b)
            lines.append(f"Geometric mean speedup of `{vm_a}` over `{vm_b}`: **{geo_mean:.2f}x**")
            lines.append("")

            ratios = table.speedup_ratio(vm_a, vm_b)
            lines.append(f"| Benchmark | Speedup ({vm_a}/{vm_b}) |")
            lines.append("|-----------|------------------------|")
            for bm_name, ratio in sorted(ratios.items()):
                lines.append(f"| {bm_name} | {ratio:.2f}x |")
            lines.append("")

        # Per-benchmark comparison
        lines.append("## Detailed Results")
        lines.append("")
        lines.append("| Benchmark | " + " | ".join(vm_names) + " |")
        lines.append("|-----------|" + "|".join(["-----"] * len(vm_names)) + "|")

        # Collect by benchmark name
        bm_data: dict[str, dict[str, ProfileResult]] = {}
        for vm_name, results in results_by_vm.items():
            for r in results:
                bm_data.setdefault(r.benchmark_name, {})[vm_name] = r

        for bm_name in sorted(bm_data.keys()):
            row_parts = []
            for vm_name in vm_names:
                r = bm_data[bm_name].get(vm_name)
                if r:
                    if r.error:
                        row_parts.append(f"ERROR")
                    else:
                        correct = "✓" if r.result_correct else "✗"
                        row_parts.append(f"{r.wall_time_ns:,}ns {correct}")
                else:
                    row_parts.append("N/A")
            lines.append(f"| {bm_name} | " + " | ".join(row_parts) + " |")

        lines.append("")
        return "\n".join(lines)
