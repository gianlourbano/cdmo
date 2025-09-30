"""
SMT solver registry for automatic registration
"""

from typing import Dict, Callable

# Global solver registry
_smt_solvers: Dict[str, Callable] = {}

def register_smt_solver(name: str):
    """Decorator to automatically register SMT solver functions"""
    def decorator(func):
        _smt_solvers[name] = func
        return func
    return decorator

def get_registered_solvers():
    """Get all registered solver names"""
    return list(_smt_solvers.keys())

def get_solver(name: str):
    """Get solver function by name"""
    return _smt_solvers.get(name)

def get_all_solvers():
    """Get the complete solver registry"""
    return _smt_solvers.copy()