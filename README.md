# Sports Tournament Scheduling (STS) Solver

This project implements solutions to the Sports Tournament Scheduling problem using multiple optimization approaches as part of the CDMO course project.

## Problem Description

The Sports Tournament Scheduling (STS) problem involves organizing a tournament with `n` teams over `n-1` weeks, where:
- Each week has `n/2` periods
- Each period has two slots (home and away)
- Every team plays every other team exactly once
- Every team plays exactly once per week
- Every team plays at most twice in the same period throughout the tournament

## Project Structure

```
cdmo-project/
├── sts_solver/           # Main Python package
│   ├── cp/              # Constraint Programming (MiniZinc)
│   ├── sat/             # SAT solving (Z3)
│   ├── smt/             # SMT solving (Z3)
│   ├── mip/             # Mixed-Integer Programming (PuLP)
│   └── utils/           # Utility functions
├── source/              # Source code organized by approach
│   ├── CP/
│   ├── SAT/
│   ├── SMT/
│   └── MIP/
├── res/                 # Results in JSON format
│   ├── CP/
│   ├── SAT/
│   ├── SMT/
│   └── MIP/
├── docs/                # Documentation
├── Dockerfile           # Docker configuration
└── pyproject.toml       # Project configuration
```

## Setup

### Using Docker (Recommended)

1. Build the Docker image:
```bash
docker build -t sts-solver .
```

2. Run the container:
```bash
docker run -it --rm -v $(pwd)/res:/app/res sts-solver
```

### Local Development

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

3. Install system dependencies (MiniZinc, Gecode, etc.)

## Usage

### Command Line Interface

Solve a single instance:
```bash
# Constraint Programming
uv run sts-solve solve 6 CP --solver gecode --timeout 300

# MIP with different formulations
uv run sts-solve solve 8 MIP --solver SCIP --timeout 300
uv run sts-solve solve 12 MIP --solver optimized-SCIP --timeout 300
uv run sts-solve solve 14 MIP --solver compact-CBC --timeout 300
uv run sts-solve solve 16 MIP --solver flow-SCIP --timeout 300
```

List available solvers and formulations:
```bash
uv run sts-solve list-solvers --solvers
```

Run benchmark:
```bash
# Single approach with specific solver
uv run sts-solve benchmark 12 --approach CP --timeout 300

# Single approach with all available solvers
uv run sts-solve benchmark 16 --approach MIP --all-solvers

# Comprehensive benchmark (ALL approaches and solvers)
uv run sts-solve comprehensive-benchmark 14 --timeout 300

# Comprehensive benchmark for specific approaches
uv run sts-solve comprehensive-benchmark 12 --approaches "CP,MIP"
```

## **Post-Benchmark Analysis**

Comprehensive analysis and validation tools:

```bash
# Analyze all results with comprehensive statistics
uv run sts-solve analyze

# Validate all solutions and show errors
uv run sts-solve validate-results

# Export analysis to different formats
uv run sts-solve analyze --format json  # For programmatic use
uv run sts-solve analyze --format csv   # For spreadsheets

# Compare solvers on specific instance
uv run sts-solve compare-instance 14
```

**Analysis Features:**
- ✅ **Solution validation** with detailed error reporting
- 📊 **Performance statistics** (timing, success rates, reliability)
- 🏆 **Solver rankings** (fastest, most reliable, best overall)
- 📈 **Instance difficulty analysis** (hardest problems to solve)
- 💡 **Recommendations** based on performance data
- 📁 **Multiple export formats** (console, JSON, CSV)

**MIP Formulation Guide:**
- **n ≤ 10**: `SCIP` or `CBC` (standard formulation)
- **n = 12-14**: `optimized-SCIP` (symmetry breaking + presolving)
- **n ≥ 16**: `compact-CBC` or `compact-SCIP` (fewer variables)
- **Experimental**: `flow-[solver]` (network-based model)

Validate solution file:
```bash
uv run sts-solve validate res/CP/6.json
```

Validate all results with official checker:
```bash
uv run sts-solve validate-all --official
```

### Docker Usage

For complete automated experiments (as required by project):
```bash
# Run all experiments automatically
./docker_run.sh
```

For individual commands:
```bash
# Solve instance with 6 teams using CP approach
docker run --rm -v $(pwd)/res:/app/res sts-solver uv run sts-solve solve 6 CP

# Run full benchmark
docker run --rm -v $(pwd)/res:/app/res sts-solver uv run sts-solve benchmark 20
```

### Automated Execution

The project includes automation scripts as required:

1. **`run_all.sh`**: Runs all models on all instances up to 20 teams
2. **`docker_run.sh`**: Complete Docker-based automation with validation

```bash
# Local execution
./run_all.sh

# Docker execution (recommended for reproducibility)
./docker_run.sh
```

## Implementation Approaches

### 1. Constraint Programming (CP)
- Uses MiniZinc with Gecode solver
- Located in `sts_solver/cp/` and `source/CP/`

### 2. SAT Solving
- Uses Z3 SAT solver
- Located in `sts_solver/sat/` and `source/SAT/`

### 3. SMT Solving
- Uses Z3 SMT solver with theories
- Located in `sts_solver/smt/` and `source/SMT/`

### 4. Mixed-Integer Programming (MIP)
- **Multiple formulations** for better scalability:
  - **Standard**: Original match-based variables
  - **Optimized**: Enhanced with symmetry breaking and aggressive presolving
  - **Compact**: Schedule-based variables (fewer variables, better for n≥12)
  - **Flow-based**: Network flow formulation (experimental)
- Uses OR-Tools (CBC, SCIP, GUROBI, CPLEX) and PuLP
- **Auto-selection**: Optimized formulation for n≥12
- Located in `sts_solver/mip/` and `source/MIP/`

## Solution Format

Solutions are saved in JSON format following the project specification:

```json
{
  "gecode": {
    "time": 42,
    "optimal": true,
    "obj": null,
    "sol": [
      [[2, 4], [5, 1], [3, 6], [3, 4], [6, 2]],
      [[5, 6], [2, 3], [4, 5], [6, 1], [1, 4]],
      [[1, 3], [4, 6], [2, 1], [5, 2], [3, 5]]
    ]
  }
}
```

## Development

Install development dependencies:
```bash
uv sync --extra dev
```

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black .
```

Type checking:
```bash
uv run mypy sts_solver
```

## Project Requirements

- Python 3.9+
- MiniZinc 2.7+
- Gecode solver
- Z3 solver
- Docker (for reproducible builds)

## Authors

CDMO Project Team - Academic Year 2024/2025