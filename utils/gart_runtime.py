from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional


def set_single_thread_env(threads: int = 1) -> None:
    threads = max(int(threads), 1)
    value = str(threads)
    os.environ.setdefault("OMP_NUM_THREADS", value)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", value)
    os.environ.setdefault("MKL_NUM_THREADS", value)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", value)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", value)


@dataclass
class GARTStudyLimits:
    max_target_evals: int = 100
    max_closed_loop_steps: int = 20
    max_solver_calls: int = 500
    max_wall_clock_seconds: float = 300.0
    max_memory_mb: Optional[float] = 4096.0
    max_governor_bisect_iters: int = 4
    max_solver_iter: int = 1000


class ResourceGuard:
    def __init__(self, limits: GARTStudyLimits):
        self.limits = limits
        self.start = time.perf_counter()
        self.solver_calls = 0
        self.target_evals = 0
        self.closed_loop_steps = 0

    def check_time(self) -> None:
        elapsed = time.perf_counter() - self.start
        if elapsed > self.limits.max_wall_clock_seconds:
            raise RuntimeError(
                f"GART resource guard stopped run: wall time {elapsed:.1f}s "
                f"exceeded limit {self.limits.max_wall_clock_seconds:.1f}s."
            )

    def check_memory(self) -> None:
        if self.limits.max_memory_mb is None:
            return
        try:
            import psutil

            process = psutil.Process()
            rss_mb = process.memory_info().rss / (1024.0 * 1024.0)
            if rss_mb > self.limits.max_memory_mb:
                raise RuntimeError(
                    f"GART resource guard stopped run: RSS memory {rss_mb:.1f} MB "
                    f"exceeded limit {self.limits.max_memory_mb:.1f} MB."
                )
        except ImportError:
            return

    def tick_solver(self, n: int = 1) -> None:
        self.solver_calls += int(n)
        if self.solver_calls > self.limits.max_solver_calls:
            raise RuntimeError(
                f"GART resource guard stopped run: solver calls {self.solver_calls} "
                f"exceeded limit {self.limits.max_solver_calls}."
            )
        self.check_time()
        self.check_memory()

    def tick_target(self, n: int = 1) -> None:
        self.target_evals += int(n)
        if self.target_evals > self.limits.max_target_evals:
            raise RuntimeError(
                f"GART resource guard stopped run: target evaluations {self.target_evals} "
                f"exceeded limit {self.limits.max_target_evals}."
            )
        self.check_time()
        self.check_memory()

    def tick_closed_loop(self, n: int = 1) -> None:
        self.closed_loop_steps += int(n)
        if self.closed_loop_steps > self.limits.max_closed_loop_steps:
            raise RuntimeError(
                f"GART resource guard stopped run: closed-loop steps {self.closed_loop_steps} "
                f"exceeded limit {self.limits.max_closed_loop_steps}."
            )
        self.check_time()
        self.check_memory()
