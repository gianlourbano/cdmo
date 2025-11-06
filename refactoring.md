# CDMO Project Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan for the Sports Tournament Scheduling (STS) solver project. The refactoring aims to improve code maintainability, reduce duplication, enhance testability, and establish consistent patterns across all solver implementations.

**Estimated Effort:** 3-5 days
**Priority:** Medium-High
**Risk Level:** Low (incremental changes with backward compatibility)

---

## Current State Analysis

### Project Structure
```
cdmo-project/
├── sts_solver/              # Main package
│   ├── cp/                  # Constraint Programming
│   ├── sat/                 # SAT solving
│   ├── smt/                 # SMT solving (7 implementations)
│   ├── mip/                 # MIP solving (9 implementations)
│   ├── utils/               # Utilities
│   └── main.py              # CLI (680 lines!)
├── source/                  # Duplicate structure?
├── res/                     # Results
└── docs/                    # Documentation
```

### Key Issues Identified

#### 🔴 **Critical Issues**
1. **Massive CLI File** - `main.py` has 680 lines with 15+ commands
2. **Code Duplication** - Similar solver patterns repeated across 20+ files
3. **Inconsistent Registry** - Only MIP uses registry pattern
4. **Missing Abstractions** - No base solver interface

#### 🟡 **Medium Priority Issues**
5. **Tight Coupling** - Direct imports between modules
6. **Hardcoded Values** - Solver names, timeouts scattered throughout
7. **Incomplete Type Hints** - ~40% of functions lack proper typing
8. **Error Handling** - Inconsistent exception handling patterns

#### 🟢 **Nice to Have**
9. **Configuration Management** - No centralized config system
10. **Testing Infrastructure** - Limited test coverage
11. **Documentation** - Missing docstrings in many functions

---

## Refactoring Goals

### Primary Objectives
- ✅ **Reduce Complexity** - Break down large files into manageable modules
- ✅ **Eliminate Duplication** - Extract common patterns into base classes
- ✅ **Improve Maintainability** - Consistent structure across all solvers
- ✅ **Enhance Testability** - Better separation of concerns

### Success Metrics
- Main CLI file < 200 lines
- Code duplication < 10%
- Test coverage > 70%
- All public functions have type hints
- All modules have docstrings

---

## Refactoring Strategy

### Phase 1: Foundation (Priority: HIGH)
**Goal:** Establish core abstractions and patterns
**Duration:** 1-2 days

### Phase 2: CLI Restructuring (Priority: HIGH)
**Goal:** Modularize command-line interface
**Duration:** 1 day

### Phase 3: Solver Unification (Priority: MEDIUM)
**Goal:** Standardize solver implementations
**Duration:** 1-2 days

### Phase 4: Configuration & Error Handling (Priority: MEDIUM)
**Goal:** Centralize configuration and improve error handling
**Duration:** 1 day

### Phase 5: Testing & Documentation (Priority: LOW)
**Goal:** Add comprehensive tests and documentation
**Duration:** Ongoing

---

## Phase 1: Foundation

### 1.1 Create Base Solver Interface

**File:** `sts_solver/base_solver.py`

```python
"""Base solver interface for all STS solver implementations"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time

from .utils.solution_format import STSSolution
from .exceptions import InvalidInstanceError, SolverTimeoutError


@dataclass
class SolverMetadata:
    """Metadata about a solver implementation"""
    name: str
    approach: str  # CP, SAT, SMT, MIP
    version: str
    supports_optimization: bool = False
    max_recommended_size: Optional[int] = None
    description: str = ""


class BaseSolver(ABC):
    """
    Abstract base class for all STS solvers.
    
    All solver implementations should inherit from this class and implement
    the solve() method. This ensures consistent interface and behavior across
    different approaches.
    """
    
    def __init__(
        self,
        n: int,
        timeout: int = 300,
        optimization: bool = False
    ):
        """
        Initialize solver with instance parameters.
        
        Args:
            n: Number of teams (must be even and >= 4)
            timeout: Maximum solving time in seconds
            optimization: Whether to solve optimization version
            
        Raises:
            InvalidInstanceError: If instance parameters are invalid
        """
        self.n = n
        self.timeout = timeout
        self.optimization = optimization
        self.start_time: Optional[float] = None
        self._validate_instance()
    
    def _validate_instance(self) -> None:
        """Validate instance parameters"""
        if self.n % 2 != 0:
            raise InvalidInstanceError("Number of teams must be even")
        if self.n < 4:
            raise InvalidInstanceError("Number of teams must be at least 4")
        if self.timeout <= 0:
            raise InvalidInstanceError("Timeout must be positive")
    
    @abstractmethod
    def _build_model(self) -> Any:
        """
        Build the solver-specific model.
        
        Returns:
            Solver-specific model object
        """
        pass
    
    @abstractmethod
    def _solve_model(self, model: Any) -> STSSolution:
        """
        Solve the built model and return solution.
        
        Args:
            model: The solver-specific model
            
        Returns:
            STSSolution object with results
        """
        pass
    
    def solve(self) -> STSSolution:
        """
        Main solving method with timing and error handling.
        
        Returns:
            STSSolution object with results
        """
        self.start_time = time.time()
        
        try:
            model = self._build_model()
            solution = self._solve_model(model)
            return solution
        except Exception as e:
            elapsed = int(time.time() - self.start_time)
            return STSSolution(
                time=min(elapsed, self.timeout),
                optimal=False,
                obj=None,
                sol=[]
            )
    
    @property
    def elapsed_time(self) -> int:
        """Get elapsed solving time in seconds"""
        if self.start_time is None:
            return 0
        return int(time.time() - self.start_time)
    
    @property
    def is_timeout(self) -> bool:
        """Check if timeout has been reached"""
        return self.elapsed_time >= self.timeout
    
    @classmethod
    @abstractmethod
    def get_metadata(cls) -> SolverMetadata:
        """Return metadata about this solver"""
        pass
```

**Benefits:**
- ✅ Consistent interface across all solvers
- ✅ Built-in validation and error handling
- ✅ Automatic timing
- ✅ Easy to test and mock

### 1.2 Create Unified Registry System

**File:** `sts_solver/registry.py`

```python
"""
Unified solver registry for all approaches.

Replaces the approach-specific registries with a centralized system
that manages all solver implementations.
"""

from typing import Dict, Type, List, Optional
from .base_solver import BaseSolver, SolverMetadata


class SolverRegistry:
    """
    Global registry for all solver implementations.
    
    This registry manages solver classes across all approaches (CP, SAT, SMT, MIP)
    and provides a unified interface for solver discovery and instantiation.
    """
    
    def __init__(self):
        self._solvers: Dict[str, Dict[str, Type[BaseSolver]]] = {
            'CP': {},
            'SAT': {},
            'SMT': {},
            'MIP': {}
        }
    
    def register(self, approach: str, name: str):
        """
        Decorator for registering solver classes.
        
        Usage:
            @registry.register('MIP', 'scip-compact')
            class CompactSCIPSolver(BaseSolver):
                ...
        
        Args:
            approach: Solver approach (CP, SAT, SMT, MIP)
            name: Unique solver name within the approach
        """
        def decorator(cls: Type[BaseSolver]):
            if approach not in self._solvers:
                self._solvers[approach] = {}
            
            self._solvers[approach][name] = cls
            return cls
        return decorator
    
    def get_solver(
        self,
        approach: str,
        name: str
    ) -> Optional[Type[BaseSolver]]:
        """
        Get solver class by approach and name.
        
        Args:
            approach: Solver approach
            name: Solver name
            
        Returns:
            Solver class or None if not found
        """
        return self._solvers.get(approach, {}).get(name)
    
    def list_solvers(
        self,
        approach: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        List available solvers.
        
        Args:
            approach: If provided, only list solvers for this approach
            
        Returns:
            Dictionary mapping approaches to solver names
        """
        if approach:
            return {approach: list(self._solvers.get(approach, {}).keys())}
        
        return {
            approach: list(solvers.keys())
            for approach, solvers in self._solvers.items()
        }
    
    def get_metadata(
        self,
        approach: str,
        name: str
    ) -> Optional[SolverMetadata]:
        """
        Get metadata for a specific solver.
        
        Args:
            approach: Solver approach
            name: Solver name
            
        Returns:
            SolverMetadata or None if solver not found
        """
        solver_cls = self.get_solver(approach, name)
        if solver_cls:
            return solver_cls.get_metadata()
        return None
    
    def get_all_metadata(
        self,
        approach: Optional[str] = None
    ) -> Dict[str, Dict[str, SolverMetadata]]:
        """
        Get metadata for all registered solvers.
        
        Args:
            approach: If provided, only return metadata for this approach
            
        Returns:
            Nested dict: {approach: {solver_name: metadata}}
        """
        result = {}
        
        approaches = [approach] if approach else self._solvers.keys()
        
        for app in approaches:
            result[app] = {}
            for name, cls in self._solvers.get(app, {}).items():
                result[app][name] = cls.get_metadata()
        
        return result
    
    def find_best_solver(
        self,
        approach: str,
        n: int,
        optimization: bool = False
    ) -> Optional[str]:
        """
        Find the best solver for given instance parameters.
        
        Args:
            approach: Solver approach
            n: Number of teams
            optimization: Whether optimization is needed
            
        Returns:
            Name of recommended solver, or None
        """
        candidates = []
        
        for name, cls in self._solvers.get(approach, {}).items():
            metadata = cls.get_metadata()
            
            # Check if solver supports optimization if needed
            if optimization and not metadata.supports_optimization:
                continue
            
            # Check if instance size is within recommendations
            if metadata.max_recommended_size and n > metadata.max_recommended_size:
                continue
            
            candidates.append(name)
        
        # Return first candidate (could be enhanced with scoring)
        return candidates[0] if candidates else None


# Global registry instance
registry = SolverRegistry()
```

**Benefits:**
- ✅ Single source of truth for all solvers
- ✅ Easy solver discovery
- ✅ Metadata management
- ✅ Best solver selection logic

### 1.3 Create Custom Exceptions

**File:** `sts_solver/exceptions.py`

```python
"""Custom exceptions for STS solver"""


class STSException(Exception):
    """Base exception for STS solver"""
    pass


class InvalidInstanceError(STSException):
    """Raised when instance parameters are invalid"""
    pass


class SolverNotFoundError(STSException):
    """Raised when requested solver is not available"""
    pass


class SolverTimeoutError(STSException):
    """Raised when solver exceeds timeout"""
    pass


class InvalidSolutionError(STSException):
    """Raised when solution validation fails"""
    pass


class ConfigurationError(STSException):
    """Raised when configuration is invalid"""
    pass


class BuildError(STSException):
    """Raised when model building fails"""
    pass
```

---

## Phase 2: CLI Restructuring

### 2.1 New CLI Structure

**Goal:** Break down 680-line `main.py` into focused, maintainable modules

**New Structure:**
```
sts_solver/cli/
├── __init__.py              # Exports main cli() function
├── main.py                  # Main entry point (~50 lines)
├── commands/
│   ├── __init__.py
│   ├── solve.py             # solve command
│   ├── benchmark.py         # benchmark & comprehensive-benchmark
│   ├── validate.py          # validate, validate-all, validate-results
│   ├── analyze.py           # analyze, compare-instance
│   ├── clean.py             # clean-results, remove-solver
│   ├── list_solvers.py      # list-solvers
│   └── backup.py            # backup-results
└── utils.py                 # Shared CLI utilities
```

### 2.2 Main Entry Point

**File:** `sts_solver/cli/main.py`

```python
"""Main CLI entry point"""

import click
from .commands import (
    solve,
    benchmark,
    validate,
    analyze,
    clean,
    list_solvers,
    backup
)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Sports Tournament Scheduling (STS) Problem Solver"""
    pass


# Register commands
cli.add_command(solve.solve)
cli.add_command(solve.solve_command)

cli.add_command(benchmark.benchmark)
cli.add_command(benchmark.comprehensive_benchmark)

cli.add_command(validate.validate)
cli.add_command(validate.validate_all)
cli.add_command(validate.validate_results)

cli.add_command(analyze.analyze)
cli.add_command(analyze.compare_instance)

cli.add_command(clean.clean_results)
cli.add_command(clean.remove_solver)

cli.add_command(list_solvers.list_solvers)

cli.add_command(backup.backup_results_cmd)


if __name__ == "__main__":
    cli()
```

### 2.3 Example Command Module

**File:** `sts_solver/cli/commands/solve.py`

```python
"""Solve command implementation"""

import click
from pathlib import Path
import time
from typing import Optional

from ...registry import registry
from ...exceptions import InvalidInstanceError, SolverNotFoundError
from ...utils.solution_format import save_results
from ..utils import validate_instance_size


@click.command()
@click.argument("n", type=int)
@click.argument("approach", type=click.Choice(["CP", "SAT", "SMT", "MIP"]))
@click.option("--solver", "-s", help="Specific solver to use")
@click.option("--timeout", "-t", default=300, help="Timeout in seconds")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--optimization", is_flag=True, help="Enable optimization")
@click.option("--name", help="Custom name for results")
def solve(
    n: int,
    approach: str,
    solver: Optional[str],
    timeout: int,
    output: Optional[str],
    optimization: bool,
    name: Optional[str]
):
    """Solve STS instance with N teams using specified approach"""
    
    # Validate instance
    try:
        validate_instance_size(n)
    except InvalidInstanceError as e:
        click.echo(f"Error: {e}", err=True)
        return 1
    
    # Setup output directory
    output_dir = Path(output) if output else Path("res") / approach
    
    # Display info
    click.echo(f"Solving STS instance with {n} teams using {approach} approach")
    if solver:
        click.echo(f"Using solver: {solver}")
    
    # Get solver class
    solver_name = solver or registry.find_best_solver(approach, n, optimization)
    if not solver_name:
        click.echo(f"Error: No suitable solver found for {approach}", err=True)
        return 1
    
    solver_cls = registry.get_solver(approach, solver_name)
    if not solver_cls:
        click.echo(f"Error: Solver '{solver_name}' not found", err=True)
        return 1
    
    # Solve
    start_time = time.time()
    try:
        solver_instance = solver_cls(n, timeout, optimization)
        result = solver_instance.solve()
        
        # Save results
        result_name = name or solver_name
        results = {result_name: result}
        save_results(n, approach, results, output_dir)
        
        click.echo(f"Solution completed in {time.time() - start_time:.2f}s")
        click.echo(f"Results saved to {output_dir / f'{n}.json'}")
        
        return 0
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1
```

**Benefits:**
- ✅ Each command is self-contained (~50-100 lines)
- ✅ Easy to test individual commands
- ✅ Clear separation of concerns
- ✅ Easier to add new commands

---

## Phase 3: Solver Unification

### 3.1 Refactor MIP Solvers

**Current Issue:** 9 different MIP implementations with lots of duplication

**Strategy:**
1. Create `MIPBaseSolver` extending `BaseSolver`
2. Extract common MIP patterns (variable creation, constraints)
3. Use strategy pattern for formulations

**File:** `sts_solver/mip/base.py`

```python
"""Base classes for MIP solvers"""

from abc import abstractmethod
from typing import Any, List, Tuple
from ortools.sat.python import cp_model

from ..base_solver import BaseSolver, SolverMetadata


class MIPBaseSolver(BaseSolver):
    """
    Base class for all MIP solvers.
    
    Provides common functionality for Mixed-Integer Programming approaches.
    """
    
    def __init__(
        self,
        n: int,
        backend: str,
        timeout: int = 300,
        optimization: bool = False
    ):
        super().__init__(n, timeout, optimization)
        self.backend = backend
        self.weeks = n - 1
        self.periods = n // 2
    
    @abstractmethod
    def _create_variables(self, model: Any) -> dict:
        """Create decision variables for the model"""
        pass
    
    @abstractmethod
    def _add_constraints(self, model: Any, variables: dict) -> None:
        """Add problem constraints to the model"""
        pass
    
    def _add_common_constraints(self, model: Any, variables: dict) -> None:
        """Add constraints common to all MIP formulations"""
        # Implementation of common constraints
        pass
    
    def _extract_solution(
        self,
        model: Any,
        variables: dict,
        status: Any
    ) -> List[List[List[int]]]:
        """Extract solution from solved model"""
        # Common solution extraction logic
        pass
```

**Refactored Solver Example:**

```python
"""Compact MIP formulation"""

from .base import MIPBaseSolver
from ..base_solver import SolverMetadata
from ...registry import registry


@registry.register('MIP', 'compact-scip')
class CompactSCIPSolver(MIPBaseSolver):
    """Compact MIP formulation using SCIP backend"""
    
    def __init__(self, n: int, timeout: int = 300, optimization: bool = False):
        super().__init__(n, 'SCIP', timeout, optimization)
    
    def _build_model(self):
        # Use parent's common functionality
        model = self._create_ortools_model()
        variables = self._create_variables(model)
        self._add_constraints(model, variables)
        return (model, variables)
    
    def _create_variables(self, model):
        """Create schedule-based variables (compact formulation)"""
        # Compact formulation specific variables
        # ...
    
    def _add_constraints(self, model, variables):
        """Add compact formulation constraints"""
        self._add_common_constraints(model, variables)
        # Add compact-specific constraints
        # ...
    
    def _solve_model(self, model_data):
        model, variables = model_data
        # Solve using SCIP
        # ...
    
    @classmethod
    def get_metadata(cls) -> SolverMetadata:
        return SolverMetadata(
            name="compact-scip",
            approach="MIP",
            version="1.0",
            supports_optimization=True,
            max_recommended_size=16,
            description="Compact MIP formulation with SCIP backend"
        )
```

### 3.2 Refactor SMT Solvers

**Current Issue:** 7 SMT implementations with similar structure

**Strategy:** Similar to MIP, create base class and use composition

**File:** `sts_solver/smt/base.py`

```python
"""Base classes for SMT solvers"""

from abc import abstractmethod
from z3 import *

from ..base_solver import BaseSolver


class SMTBaseSolver(BaseSolver):
    """Base class for all SMT solvers"""
    
    def __init__(
        self,
        n: int,
        timeout: int = 300,
        optimization: bool = False,
        use_tactics: bool = False
    ):
        super().__init__(n, timeout, optimization)
        self.use_tactics = use_tactics
        self.weeks = n - 1
        self.periods = n // 2
    
    def _build_model(self):
        """Build Z3 solver with constraints"""
        solver = self._create_solver()
        variables = self._create_variables()
        self._add_constraints(solver, variables)
        return (solver, variables)
    
    def _create_solver(self):
        """Create Z3 solver with appropriate configuration"""
        if self.use_tactics:
            return self._create_tactical_solver()
        else:
            solver = Solver()
            solver.set("timeout", self.timeout * 1000)
            return solver
    
    @abstractmethod
    def _create_variables(self):
        """Create Z3 variables"""
        pass
    
    @abstractmethod  
    def _add_constraints(self, solver, variables):
        """Add SMT constraints"""
        pass
    
    def _create_tactical_solver(self):
        """Create solver with tactics for better performance"""
        # Common tactical configuration
        pass
```

### 3.3 Solver Migration Checklist

For each solver implementation:

- [ ] Inherit from appropriate base class
- [ ] Implement required abstract methods
- [ ] Register with unified registry
- [ ] Add metadata with `get_metadata()`
- [ ] Remove duplicated code
- [ ] Add docstrings
- [ ] Add type hints
- [ ] Write unit tests

**Priority Order:**
1. MIP solvers (most duplication)
2. SMT solvers (7 implementations)
3. SAT solvers
4. CP solvers

---

## Phase 4: Configuration & Error Handling

### 4.1 Configuration System

**File:** `sts_solver/config.py`

```python
"""Configuration management for STS solver"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path
import json


@dataclass
class SolverConfig:
    """Configuration for a specific solver"""
    name: str
    timeout: int = 300
    enabled: bool = True
    parameters: Dict[str, any] = field(default_factory=dict)


@dataclass
class Config:
    """
    Global configuration for STS solver.
    
    Can be loaded from file or environment variables.
    """
    
    # Directories
    results_dir: Path = Path("res")
    docs_dir: Path = Path("docs")
    
    # Default settings
    default_timeout: int = 300
    default_approach: str = "CP"
    
    # Solver configurations
    solver_configs: Dict[str, SolverConfig] = field(default_factory=dict)
    
    # Benchmark settings
    benchmark_instances: List[int] = field(default_factory=lambda: list(range(4, 21, 2)))
    benchmark_timeout: int = 300
    
    # Validation settings
    validate_solutions: bool = True
    official_checker_path: Optional[Path] = None
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    @classmethod
    def from_file(cls, path: Path) -> 'Config':
        """Load configuration from JSON file"""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        import os
        config = cls()
        
        if timeout := os.getenv('STS_DEFAULT_TIMEOUT'):
            config.default_timeout = int(timeout)
        
        if results_dir := os.getenv('STS_RESULTS_DIR'):
            config.results_dir = Path(results_dir)
        
        return config
    
    def save(self, path: Path) -> None:
        """Save configuration to file"""
        data = {
            'results_dir': str(self.results_dir),
            'default_timeout': self.default_timeout,
            # ... other fields
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_solver_config(self, solver_name: str) -> Optional[SolverConfig]:
        """Get configuration for specific solver"""
        return self.solver_configs.get(solver_name)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        # Try to load from default locations
        config_paths = [
            Path('sts_config.json'),
            Path.home() / '.sts_config.json',
        ]
        
        for path in config_paths:
            if path.exists():
                _config = Config.from_file(path)
                break
        else:
            # Load from environment or use defaults
            _config = Config.from_env()
    
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance"""
    global _config
    _config = config
```

### 4.2 Enhanced Error Handling

**Strategy:** Add context managers and error recovery

**File:** `sts_solver/utils/error_handling.py`

```python
"""Error handling utilities"""

from contextlib import contextmanager
from typing import Optional, Callable
import logging

from ..exceptions import STSException, SolverTimeoutError


logger = logging.getLogger(__name__)


@contextmanager
def handle_solver_errors(
    solver_name: str,
    instance_size: int,
    timeout: int
):
    """
    Context manager for consistent solver error handling.
    
    Usage:
        with handle_solver_errors('SCIP', 12, 300):
            result = solver.solve()
    """
    try:
        yield
    except SolverTimeoutError:
        logger.warning(f"{solver_name} timed out on instance {instance_size}")
        raise
    except STSException as e:
        logger.error(f"{solver_name} error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in {solver_name}: {e}")
        raise STSException(f"Solver failed: {e}") from e


def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying failed operations.
    
    Usage:
        @retry_on_failure(max_attempts=3)
        def unstable_operation():
            ...
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            import time
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
        
        return wrapper
    return decorator
```

---

## Phase 5: Testing & Documentation

### 5.1 Testing Infrastructure

**New Structure:**
```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── unit/
│   ├── test_base_solver.py
│   ├── test_registry.py
│   ├── test_config.py
│   ├── test_cp_solvers.py
│   ├── test_sat_solvers.py
│   ├── test_smt_solvers.py
│   └── test_mip_solvers.py
├── integration/
│   ├── test_cli.py
│   ├── test_benchmark.py
│   └── test_validation.py
└── fixtures/
    ├── solutions/           # Example solutions
    └── configs/             # Test configurations
```

**Example Test:**

```python
"""tests/unit/test_base_solver.py"""

import pytest
from sts_solver.base_solver import BaseSolver, SolverMetadata
from sts_solver.exceptions import InvalidInstanceError
from sts_solver.utils.solution_format import STSSolution


class DummySolver(BaseSolver):
    """Dummy solver for testing"""
    
    def _build_model(self):
        return None
    
    def _solve_model(self, model):
        return STSSolution(time=1, optimal=True, obj=None, sol=[])
    
    @classmethod
    def get_metadata(cls):
        return SolverMetadata(
            name="dummy",
            approach="TEST",
            version="1.0"
        )


class TestBaseSolver:
    """Tests for BaseSolver"""
    
    def test_valid_instance(self):
        """Test solver with valid parameters"""
        solver = DummySolver(n=6, timeout=10)
        assert solver.n == 6
        assert solver.timeout == 10
    
    def test_odd_teams_raises_error(self):
        """Test that odd number of teams raises error"""
        with pytest.raises(InvalidInstanceError):
            DummySolver(n=5)
    
    def test_too_few_teams_raises_error(self):
        """Test that too few teams raises error"""
        with pytest.raises(InvalidInstanceError):
            DummySolver(n=2)
    
    def test_solve_returns_solution(self):
        """Test that solve() returns a solution"""
        solver = DummySolver(n=6)
        result = solver.solve()
        assert isinstance(result, STSSolution)
        assert result.time >= 0
    
    def test_elapsed_time(self):
        """Test elapsed time tracking"""
        solver = DummySolver(n=6)
        solver.solve()
        assert solver.elapsed_time >= 0
```

### 5.2 Documentation Standards

**Docstring Template:**

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """
    Short one-line description.
    
    Longer description explaining the function's purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When this exception is raised
    
    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        expected_output
    
    Note:
        Additional notes or warnings
    """
```

---

## Implementation Roadmap

### Week 1: Foundation & CLI

**Day 1-2: Foundation**
- [ ] Create `base_solver.py` with `BaseSolver` class
- [ ] Create unified `registry.py`
- [ ] Create `exceptions.py`
- [ ] Update `__init__.py` files to export new classes
- [ ] Write unit tests for foundation classes

**Day 3: CLI Restructuring**
- [ ] Create new `cli/` directory structure
- [ ] Split `main.py` into command modules
- [ ] Update imports throughout project
- [ ] Test all CLI commands

### Week 2: Solver Unification

**Day 4-5: MIP Refactoring**
- [ ] Create `MIPBaseSolver` base class
- [ ] Refactor `ortools_solver.py` to use base class
- [ ] Refactor compact formulations
- [ ] Refactor optimized formulations
- [ ] Update all MIP solvers to register with unified registry
- [ ] Write tests for MIP solvers

**Day 6: SMT/SAT Refactoring**
- [ ] Create `SMTBaseSolver` base class
- [ ] Refactor SMT solvers
- [ ] Refactor SAT solvers
- [ ] Register all with unified registry
- [ ] Write tests

### Week 3: Configuration & Polish

**Day 7: Configuration & Error Handling**
- [ ] Implement configuration system
- [ ] Add error handling utilities
- [ ] Update all solvers to use new error handling
- [ ] Add logging throughout

**Day 8-9: Testing & Documentation**
- [ ] Write comprehensive unit tests (target 70%+ coverage)
- [ ] Write integration tests
- [ ] Add docstrings to all public functions
- [ ] Update README.md
- [ ] Create migration guide

**Day 10: Final Review**
- [ ] Code review
- [ ] Performance testing
- [ ] Update CHANGELOG.md
- [ ] Create release notes

---

## Migration Strategy

### Backward Compatibility

During refactoring, maintain backward compatibility:

1. **Keep old interfaces** - Don't delete old functions immediately
2. **Add deprecation warnings** - Warn users about deprecated functions
3. **Gradual migration** - Move functionality incrementally
4. **Keep tests passing** - Ensure all existing tests pass after each change

**Example Deprecation:**

```python
import warnings

def solve_mip_old_function(n, solver):
    """Old function - DEPRECATED"""
    warnings.warn(
        "solve_mip_old_function is deprecated, use registry.get_solver instead",
        DeprecationWarning,
        stacklevel=2
    )
    # Call new implementation
    return solve_mip_new(n, solver)
```

### Testing Strategy

1. **Before refactoring:** Run full test suite and save results as baseline
2. **During refactoring:** Run tests after each module change
3. **After refactoring:** Run full integration tests
4. **Benchmark:** Compare performance before/after

---

## Risk Assessment

### Low Risk Changes ✅
- Adding new base classes
- Creating new utility modules
- Adding documentation
- Adding tests

### Medium Risk Changes ⚠️
- Splitting CLI into modules
- Refactoring individual solver implementations
- Changing import paths

### High Risk Changes 🔴
- Removing old code
- Changing public APIs
- Major restructuring

**Mitigation:**
- Create feature branches
- Review all changes
- Maintain test coverage
- Document breaking changes

---

## Success Criteria

### Code Quality Metrics

- [ ] Main CLI file < 200 lines
- [ ] No function > 50 lines
- [ ] No file > 500 lines
- [ ] Code duplication < 10%
- [ ] Test coverage > 70%
- [ ] All public functions have docstrings
- [ ] All public functions have type hints
- [ ] No pylint warnings

### Functional Requirements

- [ ] All existing tests pass
- [ ] All CLI commands work
- [ ] All solvers produce correct solutions
- [ ] Performance not degraded
- [ ] Backward compatibility maintained (with deprecation warnings)

### Documentation

- [ ] All modules have module docstrings
- [ ] All classes have class docstrings
- [ ] All public functions have docstrings
- [ ] README.md updated
- [ ] Migration guide created
- [ ] API reference generated

---

## Monitoring & Rollback Plan

### Monitoring

During refactoring, monitor:
- Test pass rate
- Performance benchmarks
- Code coverage
- Import errors
- Deprecation warnings

### Rollback Plan

If issues arise:

1. **Minor issues:** Fix forward
2. **Major issues:** Revert to last stable commit
3. **Critical issues:** Emergency rollback to tagged release

Keep these git tags during migration:
- `pre-refactor-baseline` - Before refactoring starts
- `phase-1-complete` - After foundation phase
- `phase-2-complete` - After CLI restructuring
- `phase-3-complete` - After solver unification
- `refactor-complete` - After all phases done

---

## Post-Refactoring

### Maintenance

After refactoring:
- Update CI/CD pipelines
- Update documentation
- Train team on new structure
- Archive old documentation

### Future Improvements

Consider for future iterations:
- Add more solver backends (Gurobi, CPLEX Pro features)
- Implement parallel solving
- Add web UI for results visualization
- Create solver comparison dashboard
- Add automated performance regression testing

---

## Conclusion

This refactoring plan provides a systematic approach to improving the CDMO project's codebase. By following the phased approach and maintaining backward compatibility, we can significantly improve code quality while minimizing risk.

**Key Benefits:**
- 🎯 **Reduced Complexity** - Smaller, focused modules
- ♻️ **Less Duplication** - Shared base classes and utilities
- 🧪 **Better Testability** - Clear interfaces and dependencies
- 📚 **Improved Documentation** - Consistent docstrings
- 🔧 **Easier Maintenance** - Logical organization and patterns

**Estimated Timeline:** 2-3 weeks
**Estimated Effort:** 60-80 hours
**Risk Level:** Low-Medium (with proper testing and incremental approach)

---

*Last Updated: [Current Date]*
*Author: Refactoring Team*
*Version: 1.0*
