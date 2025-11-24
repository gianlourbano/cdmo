"""
Unified solver registry for all approaches (CP, SAT, SMT, MIP).
"""

from typing import Dict, Type, List, Optional
from .base_solver import BaseSolver, SolverMetadata


class SolverRegistry:
    def __init__(self):
        self._solvers: Dict[str, Dict[str, Type[BaseSolver]]] = {
            'CP': {},
            'SAT': {},
            'SMT': {},
            'MIP': {}
        }

    def register(self, approach: str, name: str):
        def decorator(cls: Type[BaseSolver]):
            if approach not in self._solvers:
                self._solvers[approach] = {}
            self._solvers[approach][name] = cls
            return cls
        return decorator

    def get_solver(self, approach: str, name: str) -> Optional[Type[BaseSolver]]:
        return self._solvers.get(approach, {}).get(name)

    def list_solvers(self, approach: Optional[str] = None) -> Dict[str, List[str]]:
        if approach:
            return {approach: list(self._solvers.get(approach, {}).keys())}
        return {a: list(solvers.keys()) for a, solvers in self._solvers.items()}

    def get_metadata(self, approach: str, name: str) -> Optional[SolverMetadata]:
        cls = self.get_solver(approach, name)
        return cls.get_metadata() if cls else None

    def get_all_metadata(self, approach: Optional[str] = None) -> Dict[str, Dict[str, SolverMetadata]]:
        result: Dict[str, Dict[str, SolverMetadata]] = {}
        items = [(approach, self._solvers.get(approach, {}))] if approach else self._solvers.items()
        for a, solvers in items:
            result[a] = {name: cls.get_metadata() for name, cls in solvers.items()}
        return result

    def find_best_solver(self, approach: str, n: int, optimization: bool = False) -> Optional[str]:
        print("CALLED THIS UGLY MF")
        names: List[str] = []
        for name, cls in self._solvers.get(approach, {}).items():
            md = cls.get_metadata()
            if optimization and not md.supports_optimization:
                continue
            if md.max_recommended_size and n > md.max_recommended_size:
                continue
            names.append(name)
        return names[0] if names else None


# Global registry instance
registry = SolverRegistry()
