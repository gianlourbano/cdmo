"""
MIP solver registry for automatic registration
"""

from typing import Dict, Callable

# Global solver registry
_mip_solvers: Dict[str, Callable] = {}

def register_mip_solver(name: str):
    """Decorator to automatically register MIP solver functions"""
    def decorator(func):
        _mip_solvers[name] = func
        return func
    return decorator

def get_registered_solvers():
    """Get all registered solver names"""
    return list(_mip_solvers.keys())

def get_solver(name: str):
    """Get solver function by name"""
    return _mip_solvers.get(name)

def get_all_solvers():
    """Get the complete solver registry"""
    return _mip_solvers.copy()