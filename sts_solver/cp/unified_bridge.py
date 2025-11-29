"""CP unified registry bridge: expose all MiniZinc .mzn models from source/CP.

Backends (gecode/chuffed) are selected via --solver. Models are registered by
their filename stem (without .mzn). Optimization support is inferred from
filename prefix 'opt_'.
"""
from typing import Any
from pathlib import Path

from ..base_solver import BaseSolver, SolverMetadata
from ..registry import registry
from ..utils.solution_format import STSSolution
from .solver import solve_cp_mzn


CP_DIR = Path(__file__).parent.parent.parent / "source" / "CP"


def _supports_opt_from_name(stem: str) -> bool:
    return stem.lower().startswith("opt_")


def _register_cp_models() -> None:
    if not CP_DIR.exists():
        return
    for p in sorted(CP_DIR.glob("*.mzn")):
        stem = p.stem
        model_path = p
        supports_opt = _supports_opt_from_name(stem)

        # Dynamically build a solver class for this model
        class _CPDynamicModel(BaseSolver):
            _model_path = model_path
            _name = stem

            def _build_model(self) -> Any:
                return None

            def _solve_model(self, model: Any) -> STSSolution:
                backend = getattr(self, "backend", "gecode")
                return solve_cp_mzn(self._model_path, self.n, backend, self.timeout, search_strategy=None)

            @classmethod
            def get_metadata(cls) -> SolverMetadata:
                return SolverMetadata(
                    name=cls._name,
                    approach="CP",
                    version="1.0",
                    supports_optimization=supports_opt,
                    description=f"MiniZinc model: {cls._name}.mzn",
                )

        # Register using the stem as the model name
        registry.register("CP", stem)(_CPDynamicModel)


_register_cp_models()
