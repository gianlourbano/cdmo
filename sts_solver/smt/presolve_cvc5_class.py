"""Class-based SMT presolve using cvc5 model import from SMT2 file."""

from typing import Any, Dict, List
import cvc5  # type: ignore

from ..utils.solution_format import STSSolution
from .base import SMTBaseSolver
from ..base_solver import SolverMetadata


class SMTPresolveCVC5NativeSolver(SMTBaseSolver):
    def _build_model(self) -> Dict[str, Any]:
        return {"n": self.n}

    def _solve_model(self, model: Dict[str, Any]) -> STSSolution:
        n: int = model["n"]
        src_folder = "source/SMT"
        input_file = f"{src_folder}/from_z3__{n}.smt2"

        solver = cvc5.Solver()  # type: ignore[attr-defined]
        solver.setOption("incremental", "true")
        solver.setOption("produce-models", "true")

        parser = cvc5.InputParser(solver)  # type: ignore[attr-defined]
        file_contents = open(input_file, "r").read()
        parser.setStringInput(cvc5.InputLanguage.SMT_LIB_2_6, file_contents, "input.smt2")  # type: ignore[attr-defined]
        sm = parser.getSymbolManager()

        model_node = None
        while True:
            cmd = parser.nextCommand()
            if cmd.isNull():
                break
            if cmd.getCommandName() == "get-model":
                model_node = cmd.invoke(solver, sm)
            else:
                cmd.invoke(solver, sm)

        if model_node is not None:
            lines = str(model_node).splitlines()
            periods: List[List[List[int]]] = []
            for line in lines:
                if line.startswith("(define-fun p_"):
                    parts = line.split()
                    var_name = parts[1]
                    try:
                        value = int(parts[-1].strip(")"))
                    except Exception:
                        continue
                    _, w, i, j = var_name.split("_")
                    w_idx = int(w) - 1
                    i_t = int(i)
                    j_t = int(j)
                    period_idx = value - 1
                    while len(periods) <= period_idx:
                        periods.append([[] for _ in range(n - 1)])
                    periods[period_idx][w_idx] = [i_t, j_t]

            obj = None
            if self.optimization:
                home_counts = [0] * (n + 1)
                away_counts = [0] * (n + 1)
                for period_games in periods:
                    for home, away in period_games:
                        if home and away:
                            home_counts[home] += 1
                            away_counts[away] += 1
                obj = sum(abs(home_counts[t] - away_counts[t]) for t in range(1, n + 1))

            return STSSolution(time=0, optimal=True, obj=obj, sol=periods)

        return STSSolution(time=0, optimal=False, obj=None, sol=[])

    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="presolve_cvc5",
            approach="SMT",
            version="1.0",
            supports_optimization=True,
            description="cvc5-based presolve reading SMT2 model",
        )
