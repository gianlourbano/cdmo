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
import tempfile
import re

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
    """Build schedule periods x weeks from period/home matrices.

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


def _rewrite_solve_line(model_text: str, strategy: str, optimization: bool) -> str:
    """Inject a chosen search strategy by replacing the 'solve' line.

    Strategies supported (case-insensitive):
    - ff_luby: first_fail, indomain_min + restart_luby(50)
    - ff_geometric: first_fail, indomain_min + restart_geometric(50)
    - plain: remove custom search annotations (just 'solve satisfy' or 'solve minimize total_imbalance')
    - raw:<annotation>: inject raw annotation after 'solve ::'

    If optimization is True or model declares minimize total_imbalance, we keep 'minimize total_imbalance'; otherwise 'satisfy'.
    """
    lines = model_text.splitlines()
    # Detect optimization intent from existing solve
    existing_opt = any("minimize" in l for l in lines if l.strip().startswith("solve"))
    do_opt = optimization or existing_opt

    def build(annotation: Optional[str]) -> str:
        if do_opt:
            tail = "minimize total_imbalance;"
        else:
            tail = "satisfy;"
        if annotation:
            return f"solve :: {annotation} {tail}"
        return f"solve {tail}"

    strategy_lower = strategy.lower()
    annotation = None
    if strategy_lower == "ff_luby":
        annotation = ("int_search([period[i,j] | i,j in Teams where i<j]++"
                      "[home[i,j] | i,j in Teams where i<j], first_fail, indomain_min) :: restart_luby(50)")
    elif strategy_lower == "ff_geometric":
        annotation = ("int_search([period[i,j] | i,j in Teams where i<j]++"
                      "[home[i,j] | i,j in Teams where i<j], first_fail, indomain_min) :: restart_geometric(50)")
    elif strategy_lower.startswith("raw:"):
        annotation = strategy[4:]
    elif strategy_lower == "plain":
        annotation = None
    else:
        # Unrecognized strategy -> leave original
        return model_text

    solve_pattern = re.compile(r"^\s*solve.*;\s*$")
    replaced = False
    for idx, line in enumerate(lines):
        if solve_pattern.match(line):
            lines[idx] = build(annotation)
            replaced = True
            break
    if not replaced:
        # Append solve line if missing
        lines.append(build(annotation))
    return "\n".join(lines) + "\n"


def solve_cp_mzn(
    model_path: Path,
    n: int,
    backend: str = "gecode",
    timeout: int = 300,
    search_strategy: Optional[str] = None,
) -> STSSolution:
    """Run a specific MiniZinc model file and parse into STSSolution.

    Parsing strategy:
    - If 'schedule' (flat) is present: reshape to periods x weeks
    - Else if 'period' and 'home' are present: reconstruct from circle weeks
    - Objective: read 'total_imbalance' if present, else None
    """
    start_time = time.time()
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"MiniZinc model not found at {model_path}")
        data: Dict[str, Any] = {
            "n": n,
            "num_teams": n,
            "num_weeks": n - 1,
            "num_periods": n // 2,
        }
        # Provide week matrix for models that might expect it
        try:
            data["week"] = _circle_weeks(n)
        except Exception:
            pass

        effective_path = model_path
        if search_strategy:
            try:
                original = model_path.read_text()
                rewritten = _rewrite_solve_line(original, search_strategy, False)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mzn")
                tmp.write(rewritten.encode("utf-8"))
                tmp.flush()
                effective_path = Path(tmp.name)
            except Exception:
                pass  # fallback to original

        result = pymzn.minizinc(
            str(effective_path),
            data=data,
            solver=backend,
            timeout=timeout,
            output_mode="dict",
        )
        elapsed = int(time.time() - start_time)
        if not result:
            return STSSolution(time=elapsed if elapsed <= timeout else timeout, optimal=False, obj=None, sol=[])

        r0 = result[0]
        obj = r0.get("total_imbalance") if isinstance(r0, dict) else None
        periods = n // 2
        weeks = n - 1

        # Case 1: direct schedule (flat list of length weeks*periods)
        schedule_flat = r0.get("schedule") if isinstance(r0, dict) else None
        if schedule_flat:
            schedule: List[List[List[int]]] = []
            for p_idx in range(periods):
                period_games: List[List[int]] = []
                for w_idx in range(weeks):
                    period_games.append(schedule_flat[w_idx * periods + p_idx])
                schedule.append(period_games)
            return STSSolution(time=elapsed, optimal=True, obj=obj, sol=schedule)

        # Case 2: period + home matrices with circle week mapping
        period_raw = r0.get("period") if isinstance(r0, dict) else None
        home_raw = r0.get("home") if isinstance(r0, dict) else None
        if period_raw is not None and home_raw is not None:
            week_matrix = _circle_weeks(n)
            schedule = _reconstruct_schedule_from_period_home(n, week_matrix, period_raw, home_raw)
            return STSSolution(time=elapsed, optimal=True, obj=obj, sol=schedule)

        # Fallback: unable to parse
        return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])
    except Exception:
        elapsed = int(time.time() - start_time)
        if elapsed > timeout:
            elapsed = timeout
        return STSSolution(time=elapsed, optimal=False, obj=None, sol=[])