"""Class-based compact SMT solver for STS (variable-reduced formulation)."""

import time
from typing import Any, Dict, Tuple
from z3 import Solver, Int, Bool, If, And, Sum, Implies, is_true

from ..utils.solution_format import STSSolution
from .base import SMTBaseSolver
from ..base_solver import SolverMetadata


class SMTCompactNativeSolver(SMTBaseSolver):
    """Compact SMT schedule formulation (class-based variable-reduced model)."""

    def _build_model(self) -> Tuple[Solver, Dict[str, Any]]:
        n = self.n
        weeks = n - 1
        periods = n // 2

        s = Solver()
        s.set("timeout", self.timeout * 1000)

        match_week: Dict[Tuple[int, int], Any] = {}
        match_period: Dict[Tuple[int, int], Any] = {}
        match_t1_home: Dict[Tuple[int, int], Any] = {}

        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                key = (t1, t2)
                wv = Int(f"week_{t1}_{t2}")
                pv = Int(f"period_{t1}_{t2}")
                hv = Bool(f"home_{t1}_{t2}")
                match_week[key] = wv
                match_period[key] = pv
                match_t1_home[key] = hv
                s.add(wv >= 0, wv < weeks, pv >= 0, pv < periods)

        # Each team plays exactly once per week
        for t in range(1, n + 1):
            for w in range(weeks):
                weekly = []
                for t2 in range(1, n + 1):
                    if t2 == t:
                        continue
                    key = (t, t2) if t < t2 else (t2, t)
                    weekly.append(If(match_week[key] == w, 1, 0))
                s.add(Sum(weekly) == 1)

        # Exactly one match per period per week
        for w in range(weeks):
            for p in range(periods):
                pmatches = []
                for t1 in range(1, n + 1):
                    for t2 in range(t1 + 1, n + 1):
                        key = (t1, t2)
                        pmatches.append(If(And(match_week[key] == w, match_period[key] == p), 1, 0))
                s.add(Sum(pmatches) == 1)

        # At most twice per team in same period
        for t in range(1, n + 1):
            for p in range(periods):
                pgames = []
                for t2 in range(1, n + 1):
                    if t2 == t:
                        continue
                    key = (t, t2) if t < t2 else (t2, t)
                    pgames.append(If(match_period[key] == p, 1, 0))
                s.add(Sum(pgames) <= 2)

        # Symmetry breaking: fix first match (1 vs 2) at week 0 period 0 home=1
        if n >= 2:
            key = (1, 2)
            s.add(match_week[key] == 0, match_period[key] == 0, match_t1_home[key] == True)

        # Additional ordering inside week 0
        ordered = []
        for t1 in range(1, n + 1):
            for t2 in range(t1 + 1, n + 1):
                ordered.append(((t1, t2), t1))
        ordered.sort(key=lambda x: x[1])
        for idx, (key, _) in enumerate(ordered[:periods]):
            if key != (1, 2):
                # Enforce ordering only for week 0 assignments (implication form)
                s.add(Implies(match_week[key] == 0, match_period[key] == idx))

        return s, {
            "match_week": match_week,
            "match_period": match_period,
            "match_t1_home": match_t1_home,
            "weeks": weeks,
            "periods": periods,
        }

    def _solve_model(self, model: Tuple[Solver, Dict[str, Any]]) -> STSSolution:
        solver, state = model
        start = time.time()
        res = solver.check()
        elapsed = int(time.time() - start)
        if res.r == 1:  # sat
            m = solver.model()
            weeks = state["weeks"]
            periods = state["periods"]
            match_week = state["match_week"]
            match_period = state["match_period"]
            match_t1_home = state["match_t1_home"]

            # Build schedule as [weeks][periods] to match other SMT variants
            schedule = [[[0, 0] for _ in range(periods)] for _ in range(weeks)]
            def _as_int(ref):
                try:
                    return ref.as_long()  # z3 IntNumRef
                except Exception:
                    try:
                        return int(str(ref))
                    except Exception:
                        return 0
            for t1 in range(1, self.n + 1):
                for t2 in range(t1 + 1, self.n + 1):
                    key = (t1, t2)
                    wv = _as_int(m.eval(match_week[key]))
                    pv = _as_int(m.eval(match_period[key]))
                    hv = m.eval(match_t1_home[key])
                    home, away = (t1, t2) if is_true(hv) else (t2, t1)
                    schedule[wv][pv] = [home, away]
            obj = None
            if self.optimization:
                home_counts = [0] * (self.n + 1)
                away_counts = [0] * (self.n + 1)
                for week_games in schedule:
                    for home, away in week_games:
                        if home and away:
                            home_counts[home] += 1
                            away_counts[away] += 1
                obj = sum(abs(home_counts[t] - away_counts[t]) for t in range(1, self.n + 1))
            return STSSolution(time=elapsed, optimal=True, obj=obj, sol=schedule)
        return STSSolution(time=min(elapsed, self.timeout), optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="compact",
            approach="SMT",
            version="1.0",
            supports_optimization=True,
            description="Compact SMT formulation (class-based)",
        )
