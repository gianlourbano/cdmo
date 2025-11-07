"""Class-based baseline SMT solver using Z3 for STS."""

import time
from typing import Any, Dict, Tuple
from z3 import Solver, Int, If, Sum, And, sat

from ..utils.solution_format import STSSolution
from .base import SMTBaseSolver
from ..base_solver import SolverMetadata


class SMTBaselineNativeSolver(SMTBaseSolver):
    """Baseline SMT formulation (class-based) replicating z3_baseline behavior."""

    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        weeks = self.weeks
        periods = self.periods
        n = self.n
        s = Solver()
        s.set("timeout", self.timeout * 1000)

        schedule: Dict[Tuple[int, int, int], Any] = {}
        for w in range(weeks):
            for p in range(periods):
                for slot in range(2):
                    var = Int(f"schedule_{w}_{p}_{slot}")
                    schedule[w, p, slot] = var
                    s.add(var >= 1, var <= n)

        # No self play
        for w in range(weeks):
            for p in range(periods):
                s.add(schedule[w, p, 0] != schedule[w, p, 1])

        # Each team exactly once per week
        for w in range(weeks):
            for t in range(1, n + 1):
                appearances = []
                for p in range(periods):
                    for slot in range(2):
                        appearances.append(If(schedule[w, p, slot] == t, 1, 0))
                s.add(Sum(appearances) == 1)

        # Each pair of teams exactly once
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                games = []
                for w in range(weeks):
                    for p in range(periods):
                        games.append(And(schedule[w, p, 0] == t1, schedule[w, p, 1] == t2))
                        games.append(And(schedule[w, p, 0] == t2, schedule[w, p, 1] == t1))
                s.add(Sum([If(g, 1, 0) for g in games]) == 1)

        # At most twice per period
        for p in range(periods):
            for t in range(1, n + 1):
                per_apps = []
                for w in range(weeks):
                    for slot in range(2):
                        per_apps.append(If(schedule[w, p, slot] == t, 1, 0))
                s.add(Sum(per_apps) <= 2)

        # Symmetry breaking: team 1 plays at home first period first week
        s.add(schedule[0, 0, 0] == 1)

        return s, {"schedule": schedule}

    def _solve_model(self, model: Tuple[Solver, Dict[str, Any]]) -> STSSolution:
        solver, state = model
        start = time.time()
        result = solver.check()
        elapsed = int(time.time() - start)
        if result == sat:
            m = solver.model()
            weeks = self.weeks
            periods = self.periods
            schedule = state["schedule"]
            sol = []
            for p in range(periods):
                period_games = []
                for w in range(weeks):
                    home_val = m.eval(schedule[w, p, 0])
                    away_val = m.eval(schedule[w, p, 1])
                    # Z3 Int values expose as_long(); fallback to pyint via int(str()) if needed
                    try:
                        home = home_val.as_long()  # type: ignore[attr-defined]
                    except Exception:
                        home = int(str(home_val))
                    try:
                        away = away_val.as_long()  # type: ignore[attr-defined]
                    except Exception:
                        away = int(str(away_val))
                    period_games.append([home, away])
                sol.append(period_games)
            obj = None if not self.optimization else 1
            return STSSolution(time=elapsed, optimal=True, obj=obj, sol=sol)
        return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="baseline",
            approach="SMT",
            version="1.0",
            supports_optimization=True,
            description="Baseline SMT formulation (class-based)",
        )
