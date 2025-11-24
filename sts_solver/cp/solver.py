"""Constraint Programming solver logic for STS using MiniZinc.

Supports two modes:
 - Satisfy only (legacy): uses `sts.mzn` model producing a flat `schedule` array
 - Optimization (home/away imbalance minimization): uses
   `circle_method_SB_modified_new_sb.mzn` and reconstructs schedule from
   `period` and `home` decision variables plus a generated circle-method
   week matrix.

Objective reported is `total_imbalance` from optimization model when present.
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import pymzn

from ..utils.solution_format import STSSolution


def _circle_weeks(n: int) -> List[List[int]]:
    """Generate week matrix for circle method.

    week[i-1][j-1] = week number (1..n-1) when i plays j, 0 on diagonal.
    """
    pivot = n
    circle = list(range(1, n))
    weeks = n - 1
    # Initialize matrix with zeros
    week: List[List[int]] = [[0 for _ in range(n)] for _ in range(n)]
    for w in range(weeks):
        # pivot vs circle[w]
        opp = circle[w]
        a, b = pivot, opp
        week[a-1][b-1] = w + 1
        week[b-1][a-1] = w + 1
        for k in range(1, n // 2):
            i = circle[(w + k) % (n - 1)]
            j = circle[(w - k) % (n - 1)]
            if i > j:
                i, j = j, i
            week[i-1][j-1] = w + 1
            week[j-1][i-1] = w + 1
    return week


def _reconstruct_schedule_from_period_home(n: int, week: List[List[int]], period: List[List[int]], home: List[List[int]]) -> List[List[List[int]]]:
    """Build schedule periods × weeks from period/home matrices.

    Returns: periods list each containing weeks list of [home, away].
    """
    weeks = n - 1
    periods = n // 2
    # Initialize structure periods x weeks
    schedule: List[List[List[int]]] = [[[] for _ in range(weeks)] for _ in range(periods)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            w = week[i - 1][j - 1]
            p = period[i - 1][j - 1]
            if w < 1 or p < 1:
                continue
            # Determine orientation: home[i-1][j-1] == 1 means i home
            if home[i - 1][j - 1] == 1:
                game = [i, j]
            else:
                game = [j, i]
            schedule[p - 1][w - 1] = game
    # Safety fill (MiniZinc should have assigned all games)
    for p in range(periods):
        for w in range(weeks):
            if not schedule[p][w]:  # fallback empty placeholder
                schedule[p][w] = [0, 0]
    return schedule


def solve_cp(
    n: int,
    solver: Optional[str] = None,
    timeout: int = 300,
    optimization: bool = False,
) -> STSSolution:
    """Solve STS using MiniZinc CP backend.

    When `optimization=True`, uses enhanced imbalance minimization model and
    returns objective value; otherwise returns satisfy-only schedule.
    """
    backend = solver or "gecode"
    start_time = time.time()

    try:
        if optimization:
            model_path = Path(__file__).parent.parent.parent / "source" / "CP" / "circle_method_SB_modified_new_sb.mzn"
            if not model_path.exists():
                raise FileNotFoundError(f"Optimization model not found at {model_path}")
            week_matrix = _circle_weeks(n)
            data = {
                "num_teams": n,
                "num_weeks": n - 1,
                "num_periods": n // 2,
                "week": week_matrix,
            }
            result = pymzn.minizinc(
                str(model_path),
                data=data,
                solver=backend,
                timeout=timeout,
                output_mode="dict",
            )
            elapsed = int(time.time() - start_time)
            if result:
                r0 = result[0]
                # Expect period and home matrices
                period_raw = r0.get("period")
                home_raw = r0.get("home")
                total_imbalance = r0.get("total_imbalance")
                if period_raw is None or home_raw is None:
                    return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])
                schedule = _reconstruct_schedule_from_period_home(n, week_matrix, period_raw, home_raw)
                return STSSolution(time=elapsed, optimal=True, obj=total_imbalance, sol=schedule)
            else:
                return STSSolution(time=timeout, optimal=False, obj=None, sol=[])
        else:
            model_path = Path(__file__).parent.parent.parent / "source" / "CP" / "sts.mzn"
            if not model_path.exists():
                raise FileNotFoundError(f"MiniZinc model not found at {model_path}")
            data = {"n": n}
            result = pymzn.minizinc(
                str(model_path),
                data=data,
                solver=backend,
                timeout=timeout,
                output_mode="dict",
            )
            elapsed = int(time.time() - start_time)
            if result:
                schedule_flat = result[0]["schedule"]
                periods = n // 2
                weeks = n - 1
                schedule: List[List[List[int]]] = []
                for p in range(periods):
                    period_games: List[List[int]] = []
                    for w in range(weeks):
                        period_games.append(schedule_flat[w * periods + p])
                    schedule.append(period_games)
                return STSSolution(time=elapsed, optimal=True, obj=None, sol=schedule)
            else:
                return STSSolution(time=timeout, optimal=False, obj=None, sol=[])
    except Exception:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            elapsed = timeout
        return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])