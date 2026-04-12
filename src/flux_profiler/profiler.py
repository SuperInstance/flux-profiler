"""Core profiling engine for FLUX VM benchmarks.

Manages benchmark execution, timing, warmup, and result collection
across multiple VM implementations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from flux_profiler.benchmarks import BenchmarkWorkload
from flux_profiler.vm_adapter import VMAdapter, ProfileResult


class Profiler:
    """Orchestrates benchmark execution across VM implementations.

    Features:
    - Warmup runs before measurement for JIT warmup / cache priming
    - Multiple iterations with nanosecond-precision timing
    - Timeout protection per execution
    - Correctness verification against expected results
    """

    def __init__(self) -> None:
        self._vms: dict[str, VMAdapter] = {}
        self._warmup_count: int = 3

    def register_vm(self, adapter: VMAdapter) -> None:
        """Register a VM adapter for benchmarking."""
        self._vms[adapter.name] = adapter

    @property
    def registered_vms(self) -> list[str]:
        """Return names of registered VMs."""
        return list(self._vms.keys())

    @property
    def warmup_count(self) -> int:
        """Number of warmup iterations before measurement."""
        return self._warmup_count

    @warmup_count.setter
    def warmup_count(self, value: int) -> None:
        """Set the number of warmup iterations."""
        self._warmup_count = max(0, value)

    def _execute_once(
        self,
        benchmark: BenchmarkWorkload,
        vm: VMAdapter,
        timeout: float,
    ) -> ProfileResult:
        """Execute a benchmark once and fill in the result metadata."""
        result = vm.execute(benchmark.bytecode, timeout=timeout)
        result.benchmark_name = benchmark.name

        # Check correctness
        if benchmark.expected_result is not None:
            result.result_correct = result.registers_final.get(0, 0) == benchmark.expected_result
        else:
            result.result_correct = True  # can't verify

        return result

    def warmup(
        self,
        benchmark: BenchmarkWorkload,
        vm_name: str,
        count: Optional[int] = None,
        timeout: float = 5.0,
    ) -> list[ProfileResult]:
        """Run warmup iterations (not included in reported results).

        Args:
            benchmark: The benchmark to warm up with.
            vm_name: Name of the registered VM to use.
            count: Number of warmup runs (default: self.warmup_count).
            timeout: Per-execution timeout in seconds.

        Returns:
            List of warmup results (discarded by callers).
        """
        if vm_name not in self._vms:
            raise ValueError(f"VM '{vm_name}' not registered. Available: {list(self._vms.keys())}")
        vm = self._vms[vm_name]
        n = count if count is not None else self._warmup_count
        results = []
        for _ in range(n):
            r = self._execute_once(benchmark, vm, timeout)
            results.append(r)
        return results

    def run_benchmark(
        self,
        benchmark: BenchmarkWorkload,
        vm_name: str,
        iterations: int = 10,
        timeout: float = 10.0,
    ) -> ProfileResult:
        """Run a single benchmark on a VM with warmup.

        Executes warmup runs first, then performs `iterations` measured runs.
        Returns the median result (by wall_time_ns).

        Args:
            benchmark: The benchmark workload to execute.
            vm_name: Name of the registered VM.
            iterations: Number of measured iterations.
            timeout: Per-execution timeout in seconds.

        Returns:
            ProfileResult with median timing from all iterations.
        """
        if vm_name not in self._vms:
            raise ValueError(f"VM '{vm_name}' not registered. Available: {list(self._vms.keys())}")

        # Warmup
        self.warmup(benchmark, vm_name, timeout=timeout)

        # Measured runs
        results = []
        vm = self._vms[vm_name]
        for _ in range(iterations):
            r = self._execute_once(benchmark, vm, timeout)
            if r.error is not None:
                # Return the errored result immediately
                return r
            results.append(r)

        if not results:
            return ProfileResult(
                benchmark_name=benchmark.name,
                vm_name=vm_name,
                wall_time_ns=0,
                instructions_executed=0,
                cycles_used=0,
                memory_reads=0,
                memory_writes=0,
                peak_stack_depth=0,
                registers_final={},
                result_correct=False,
                error="No successful iterations",
            )

        # Return median result by wall_time_ns
        results.sort(key=lambda r: r.wall_time_ns)
        return results[len(results) // 2]

    def run_suite(
        self,
        benchmarks: list[BenchmarkWorkload],
        vm_name: str,
        iterations: int = 10,
        timeout: float = 10.0,
    ) -> list[ProfileResult]:
        """Run all benchmarks on a single VM.

        Args:
            benchmarks: List of benchmarks to execute.
            vm_name: Name of the registered VM.
            iterations: Number of measured iterations per benchmark.
            timeout: Per-execution timeout.

        Returns:
            List of ProfileResults, one per benchmark.
        """
        if vm_name not in self._vms:
            raise ValueError(f"VM '{vm_name}' not registered. Available: {list(self._vms.keys())}")

        results = []
        for bm in benchmarks:
            result = self.run_benchmark(bm, vm_name, iterations, timeout)
            results.append(result)
        return results

    def run_comparative(
        self,
        benchmarks: list[BenchmarkWorkload],
        iterations: int = 10,
        timeout: float = 10.0,
    ) -> dict[str, list[ProfileResult]]:
        """Run all benchmarks across all registered VMs.

        Args:
            benchmarks: List of benchmarks to execute.
            iterations: Number of measured iterations per benchmark per VM.
            timeout: Per-execution timeout.

        Returns:
            Dict mapping VM name to list of ProfileResults.
        """
        results_by_vm: dict[str, list[ProfileResult]] = {}
        for vm_name in self._vms:
            results_by_vm[vm_name] = self.run_suite(benchmarks, vm_name, iterations, timeout)
        return results_by_vm

    def run_single(
        self,
        benchmark: BenchmarkWorkload,
        vm_name: str,
        timeout: float = 10.0,
    ) -> ProfileResult:
        """Run a single benchmark execution with no warmup or iteration.

        Useful for quick testing and debugging.

        Args:
            benchmark: The benchmark to execute.
            vm_name: Name of the registered VM.
            timeout: Per-execution timeout.

        Returns:
            ProfileResult from the single execution.
        """
        if vm_name not in self._vms:
            raise ValueError(f"VM '{vm_name}' not registered. Available: {list(self._vms.keys())}")
        return self._execute_once(benchmark, self._vms[vm_name], timeout)
