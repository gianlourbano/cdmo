# Docker Commands for STS Solver

**Project:** Sports Tournament Scheduling (STS) Solver  
**Course:** CDMO 2024/2025  
**Docker Image:** `sts-solver`

This document provides all Docker commands required by the project specification for reproducible execution.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Building the Docker Image](#building-the-docker-image)
3. [Running Individual Solvers](#running-individual-solvers)
4. [Running Benchmarks](#running-benchmarks)
5. [Validation](#validation)
6. [Analysis](#analysis)
7. [Complete Automation](#complete-automation)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Automated Execution (Recommended)

Run all experiments with a single command:

```bash
./docker_run.sh
```

This will:
1. Build the Docker image
2. Run comprehensive benchmarks on all approaches
3. Validate all solutions
4. Generate analysis reports
5. Save results to `./res/`

---

## Building the Docker Image

### Build Command

```bash
docker build -t sts-solver .
```

**What it does:**
- Installs Python 3.12 with UV package manager
- Installs all Python dependencies (Z3, OR-Tools, PuLP, etc.)
- Installs MiniZinc 2.7.5 with Gecode and Chuffed solvers
- Sets up the project structure
- Creates result directories

**Verification:**

```bash
# Verify image was built
docker images | grep sts-solver

# Check image size (should be ~1-2GB)
docker images sts-solver --format "{{.Size}}"
```

---

## Running Individual Solvers

### General Pattern

```bash
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve <N> <APPROACH> [--model NAME] [--solver BACKEND] [OPTIONS]
```

### Examples by Approach

#### Constraint Programming (CP)

```bash
# Gecode solver (decision version)
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 6 CP --solver gecode --timeout 300

# Chuffed solver (optimization version)
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 8 CP --solver chuffed --timeout 300 --optimization
```

#### SAT Solving

```bash
# Baseline SAT encoding
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 10 SAT --model baseline --timeout 300

# Pairwise SAT encoding
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 12 SAT --model pairwise --timeout 300
```

#### SMT Solving

```bash
# Presolve_2 solver (best performance)
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 14 SMT --model presolve_2 --timeout 300

# With optimization (home/away balance)
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 16 SMT --model presolve_2 --timeout 300 --optimization

# Other SMT models
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 12 SMT --model baseline --timeout 300

docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 14 SMT --model optimized --timeout 300
```

#### Mixed-Integer Programming (MIP)

```bash
# Standard formulation
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 10 MIP --model standard --solver CBC --timeout 300

# Optimized formulation (better for n≥12)
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 14 MIP --model optimized --solver CBC --timeout 300

# Compact formulation (fewer variables)
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 16 MIP --model compact --solver CBC --timeout 300

# Flow-based formulation
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 12 MIP --model flow --solver SCIP --timeout 300
```

---

## Running Benchmarks

### Single Instance, All Solvers

Test all solvers for a specific approach on one instance:

```bash
# All MIP solvers on n=12
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts benchmark 12 --approach MIP --all-solvers --timeout 300

# All SMT solvers on n=14
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts benchmark 14 --approach SMT --all-solvers --timeout 300
```

### Comprehensive Benchmark

Test all approaches and solvers up to a maximum instance size:

```bash
# Run all solvers up to n=20
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts comprehensive-benchmark 20 --timeout 300

# Run specific approaches only
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts comprehensive-benchmark 16 --approaches "CP,SMT,MIP" --timeout 300

# With optimization enabled
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts comprehensive-benchmark 14 --optimization --timeout 300
```

### Automated Script (As Required by Spec)

The project includes `run_all.sh` which runs all models on all instances:

```bash
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    /bin/bash -c "chmod +x /app/run_all.sh && /app/run_all.sh"
```

This script:
- Runs comprehensive benchmark up to n=20
- Tests all solver formulations
- Generates analysis reports
- Exports results in multiple formats

---

## Validation

### Validate Single Solution

```bash
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts validate res/CP/6.json
```

### Validate All Results

```bash
# Basic validation
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts validate-all

# With official checker (if available)
docker run --rm \
    -v $(pwd)/res:/app/res \
    -v $(pwd)/solution_checker.py:/app/solution_checker.py \
    sts-solver \
    uv run sts validate-all --official
```

### Validate and Show Errors

```bash
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts validate-results
```

---

## Analysis

### Comprehensive Analysis Report

```bash
# Console output
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts analyze

# JSON export
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts analyze --format json

# CSV export (for spreadsheets)
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts analyze --format csv
```

### Compare Solvers on Specific Instance

```bash
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts compare-instance 14
```

### List Available Models/Backends

```bash
docker run --rm sts-solver uv run sts list-models

# Filter by approach
docker run --rm sts-solver uv run sts list-models --approach MIP

# Detailed information
docker run --rm sts-solver uv run sts list-models --verbose

# Backends (CP/MIP)
docker run --rm sts-solver uv run sts list-backends
docker run --rm sts-solver uv run sts list-backends --approach CP
```

---

## Complete Automation

### Full Workflow (As Required by Spec)

This is the **recommended approach** for project submission:

```bash
./docker_run.sh
```

**What it does:**

1. **Build Phase**
   ```bash
   docker build -t sts-solver .
   ```

2. **Execution Phase**
   ```bash
   docker run --rm \
       -v $(pwd)/res:/app/res \
       -v $(pwd)/solution_checker.py:/app/solution_checker.py \
       sts-solver \
       /bin/bash -c "
           chmod +x /app/run_all.sh
           /app/run_all.sh
           uv run sts validate-all --official
       "
   ```

3. **Output**
   - All results saved to `./res/` directory
   - Organized by approach: `res/CP/`, `res/SAT/`, `res/SMT/`, `res/MIP/`
   - Analysis reports: `res/benchmark_analysis.json`, `res/benchmark_summary.csv`
   - Validation report printed to console

---

## Advanced Usage

### Interactive Shell

Enter the container for manual testing:

```bash
docker run --rm -it \
    -v $(pwd)/res:/app/res \
    sts-solver \
    /bin/bash
```

Inside the container:
```bash
# List available models/backends
uv run sts list-models
uv run sts list-backends

# Run individual solver
uv run sts solve 6 CP --solver gecode

# Run analysis
uv run sts analyze

# Exit container
exit
```

### Custom Timeout

```bash
# 10-minute timeout
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 20 SMT --model presolve_2 --timeout 600
```

### Custom Results Directory

```bash
# Save to different directory
docker run --rm -v $(pwd)/custom_results:/app/res sts-solver \
    uv run sts solve 12 MIP --model optimized --solver CBC
```

### Parallel Execution (Multiple Containers)

Run different approaches in parallel:

```bash
# Terminal 1: CP
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts benchmark 20 --approach CP --timeout 300 &

# Terminal 2: SMT
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts benchmark 20 --approach SMT --timeout 300 &

# Terminal 3: MIP
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts benchmark 20 --approach MIP --timeout 300 &

# Wait for all to complete
wait
```

---

## Troubleshooting

### Issue: Permission Denied on Results Directory

**Problem:** Docker can't write to `./res/`

**Solution:**
```bash
# Create directory with proper permissions
mkdir -p res/{CP,SAT,SMT,MIP}
chmod -R 777 res/

# Or run with user permissions
docker run --rm --user $(id -u):$(id -g) \
    -v $(pwd)/res:/app/res \
    sts-solver \
    uv run sts solve 6 CP
```

### Issue: MiniZinc Not Found

**Problem:** `minizinc: command not found`

**Solution:** Rebuild the Docker image:
```bash
docker build --no-cache -t sts-solver .
```

### Issue: Timeout Too Short

**Problem:** Solver times out before finding solution

**Solution:** Increase timeout:
```bash
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 20 SMT --solver presolve_2 --timeout 600
```

### Issue: Out of Memory

**Problem:** Docker container runs out of memory for large instances

**Solution:** Increase Docker memory limit:
```bash
docker run --rm -m 4g -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 24 MIP --solver optimized
```

### Issue: Results Not Persisting

**Problem:** Results disappear after container exits

**Solution:** Ensure volume mounting is correct:
```bash
# Use absolute path
docker run --rm -v "$(pwd)/res:/app/res" sts-solver \
    uv run sts solve 6 CP

# Verify mount
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    ls -la /app/res
```

---

## Verification Checklist

Before submitting, verify Docker execution:

- [ ] Docker image builds successfully
- [ ] Can run individual solvers for each approach (CP, SAT, SMT, MIP)
- [ ] `./docker_run.sh` completes without errors
- [ ] Results are saved to `./res/` directory
- [ ] All result files are valid JSON
- [ ] Solutions pass validation checker
- [ ] Analysis reports are generated
- [ ] Can reproduce results on different machine

---

## Project Specification Compliance

This Docker setup complies with all project requirements:

✅ **Reproducibility**: All solvers run inside Docker container  
✅ **Automation**: Single command runs all experiments (`./docker_run.sh`)  
✅ **Individual Execution**: Can run specific solver on specific instance  
✅ **Batch Execution**: Can run all solvers on all instances  
✅ **Free Software**: Uses only open-source tools (Z3, OR-Tools, MiniZinc)  
✅ **Documentation**: Complete instructions provided  
✅ **Results Format**: Saves to `res/` with correct JSON format  
✅ **Validation**: Includes solution checker integration  

---

## Quick Reference

### Most Common Commands

```bash
# 1. Build image
docker build -t sts-solver .

# 2. Run all experiments (RECOMMENDED)
./docker_run.sh

# 3. Run single solver
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts solve 12 SMT --solver presolve_2

# 4. Validate results
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts validate-all

# 5. Analyze results
docker run --rm -v $(pwd)/res:/app/res sts-solver \
    uv run sts analyze
```
