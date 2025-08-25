"""
Main CLI interface for the STS solver
"""

import click
from pathlib import Path
import json
import time
from typing import Optional, List

from .utils.solution_format import save_results, STSSolution


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Sports Tournament Scheduling (STS) Problem Solver"""
    pass


@cli.command()
@click.argument("n", type=int)
@click.argument("approach", type=click.Choice(["CP", "SAT", "SMT", "MIP"]))
@click.option("--solver", "-s", help="Specific solver to use (e.g., gecode, z3, CBC, SCIP, GUROBI) - excludes GLOP (LP-only)")
@click.option("--timeout", "-t", default=300, help="Timeout in seconds (default: 300)")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--optimization", is_flag=True, help="Enable optimization version")
@click.option("--name", help="Custom name for this approach in JSON results (default: solver name)")
def solve(n: int, approach: str, solver: Optional[str], timeout: int, output: Optional[str], optimization: bool, name: Optional[str]):
    """Solve STS instance with N teams using specified approach"""
    
    if n % 2 != 0:
        click.echo("Error: Number of teams must be even", err=True)
        return
    
    if n < 4:
        click.echo("Error: Number of teams must be at least 4", err=True)
        return
    
    output_dir = Path(output) if output else Path("res") / approach
    
    click.echo(f"Solving STS instance with {n} teams using {approach} approach")
    if solver:
        click.echo(f"Using solver: {solver}")
    
    start_time = time.time()
    
    # Import and run the appropriate solver
    try:
        if approach == "CP":
            from .cp.solver import solve_cp
            result = solve_cp(n, solver, timeout, optimization)
        elif approach == "SAT":
            from .sat.solver import solve_sat
            result = solve_sat(n, solver, timeout, optimization)
        elif approach == "SMT":
            from .smt.solver import solve_smt
            result = solve_smt(n, solver, timeout, optimization)
        elif approach == "MIP":
            from .mip.solver import solve_mip
            result = solve_mip(n, solver, timeout, optimization)
        else:
            click.echo(f"Error: Unknown approach {approach}", err=True)
            return
            
        # Save results with custom name or solver name
        result_name = name or solver or "default"
        results = {result_name: result}
        save_results(n, approach, results, output_dir)
        
        click.echo(f"Solution completed in {time.time() - start_time:.2f} seconds")
        click.echo(f"Results saved to {output_dir / f'{n}.json'}")
        
    except ImportError as e:
        click.echo(f"Error: Solver not implemented yet: {e}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--checker", "-c", type=click.Path(), help="Path to solution_checker.py")
def validate(results_file: str, checker: Optional[str]):
    """Validate solution file format and correctness"""
    
    from .utils.checker import check_solution_file
    
    try:
        # First check JSON format
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        for solver_name, solution in data.items():
            required_fields = ["time", "optimal", "obj", "sol"]
            for field in required_fields:
                if field not in solution:
                    click.echo(f"Error: Missing field '{field}' in solver '{solver_name}'", err=True)
                    return
        
        click.echo("Solution file format is valid")
        
        # Validate solution correctness
        checker_path = Path(checker) if checker else Path("solution_checker.py")
        is_valid, message = check_solution_file(Path(results_file), checker_path)
        if is_valid:
            click.echo("Solution correctness validated")
        else:
            click.echo(f"Solution validation failed: {message}", err=True)
        
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid JSON format: {e}", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.option("--checker", "-c", type=click.Path(), help="Path to solution_checker.py")
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory")
@click.option("--official", is_flag=True, help="Use official checker's directory-based validation")
def validate_all(checker: Optional[str], results_dir: str, official: bool):
    """Validate all solution files"""
    
    checker_path = Path(checker) if checker else Path("solution_checker.py")
    results_path = Path(results_dir)
    
    if official:
        # Use the official checker's directory-based approach
        from .utils.checker import validate_directory_with_official_checker
        click.echo("Using official checker validation...")
        validate_directory_with_official_checker(results_path, checker_path)
    else:
        # Use our integrated approach
        from .utils.checker import validate_all_results, print_validation_report
        click.echo("Validating all solution files...")
        validation_results = validate_all_results(results_path, checker_path)
        print_validation_report(validation_results)


@cli.command()
@click.option("--solvers", is_flag=True, help="List available solvers")
def list_solvers(solvers: bool):
    """List available solvers and formulations"""
    
    if solvers:
        from .mip.ortools_solver import get_available_ortools_solvers
        
        click.echo("=== MIP Solvers & Formulations ===")
        ortools_solvers = get_available_ortools_solvers()
        
        click.echo("Standard formulation:")
        for solver in ortools_solvers:
            click.echo(f"  - {solver}")
        
        click.echo("\nOptimized formulation (with symmetry breaking):")
        for solver in ortools_solvers:
            click.echo(f"  - optimized-{solver}")
        
        click.echo("\nCompact formulation (schedule-based variables):")
        for solver in ortools_solvers:
            click.echo(f"  - compact-{solver}")
            
        click.echo("\nFlow-based formulation (network model):")
        for solver in ortools_solvers:
            click.echo(f"  - flow-{solver}")
        
        click.echo("\nPuLP formulation (legacy):")
        click.echo("  - pulp-cbc, pulp-gurobi, pulp-cplex")
        
        click.echo("\n=== Other Approaches ===")
        click.echo("CP solvers: gecode, chuffed")
        click.echo("SAT/SMT solvers: z3")
        
        click.echo("\n=== Recommendations ===")
        click.echo("For n <= 10: Use standard formulation")
        click.echo("For n >= 12: Use optimized or compact formulation")
        click.echo("For n >= 16: Try compact or flow formulation")


@cli.command()
@click.argument("max_n", type=int)
@click.option("--approach", "-a", type=click.Choice(["CP", "SAT", "SMT", "MIP"]), default="CP")
@click.option("--solver", "-s", help="Specific solver to use")
@click.option("--timeout", "-t", default=300, help="Timeout in seconds")
@click.option("--all-solvers", is_flag=True, help="Run all available solvers for the approach")
def benchmark(max_n: int, approach: str, solver: Optional[str], timeout: int, all_solvers: bool):
    """Run benchmark for instances from 4 to MAX_N teams"""
    
    if all_solvers:
        click.echo(f"Running comprehensive benchmark for {approach} approach up to {max_n} teams")
        solvers_to_test = get_solvers_for_approach(approach)
        click.echo(f"Testing solvers: {', '.join(solvers_to_test)}")
    else:
        click.echo(f"Running benchmark for {approach} approach up to {max_n} teams")
        if solver:
            click.echo(f"Using solver: {solver}")
        solvers_to_test = [solver] if solver else [None]
    
    for n in range(4, max_n + 1, 2):  # Only even numbers
        click.echo(f"\n{'='*50}")
        click.echo(f"INSTANCE: {n} teams")
        click.echo(f"{'='*50}")
        
        for test_solver in solvers_to_test:
            if test_solver:
                click.echo(f"\nTesting {approach} with {test_solver}...")
            else:
                click.echo(f"\nTesting {approach} with default solver...")
                
            try:
                ctx = click.Context(solve)
                ctx.invoke(solve, n=n, approach=approach, solver=test_solver, timeout=timeout, name=test_solver)
            except Exception as e:
                click.echo(f"Error with {test_solver or 'default'}: {e}", err=True)


def get_solvers_for_approach(approach: str) -> List[str]:
    """Get list of solvers to test for each approach"""
    
    if approach == "CP":
        return ["gecode", "chuffed"]
    elif approach == "SAT":
        return ["z3"]
    elif approach == "SMT": 
        return ["z3"]
    elif approach == "MIP":
        from .mip.ortools_solver import get_available_ortools_solvers
        ortools_solvers = get_available_ortools_solvers()
        
        mip_solvers = []
        # Standard formulations
        for solver in ortools_solvers:
            mip_solvers.append(solver)
        
        # Optimized formulations (for larger instances)
        for solver in ortools_solvers:
            mip_solvers.append(f"optimized-{solver}")
            
        # Compact formulations (for scalability)
        for solver in ortools_solvers:
            mip_solvers.append(f"compact-{solver}")
            
        return mip_solvers
    else:
        return []


@cli.command()
@click.argument("max_n", type=int)  
@click.option("--timeout", "-t", default=300, help="Timeout in seconds")
@click.option("--approaches", help="Comma-separated list of approaches (CP,SAT,SMT,MIP)")
def comprehensive_benchmark(max_n: int, timeout: int, approaches: Optional[str]):
    """Run comprehensive benchmark across ALL approaches and solvers"""
    
    if approaches:
        approach_list = [a.strip().upper() for a in approaches.split(",")]
    else:
        approach_list = ["CP", "SAT", "SMT", "MIP"]
    
    click.echo("="*60)
    click.echo("COMPREHENSIVE STS SOLVER BENCHMARK")  
    click.echo("="*60)
    click.echo(f"Approaches: {', '.join(approach_list)}")
    click.echo(f"Instance range: 4 to {max_n} teams")
    click.echo(f"Timeout: {timeout} seconds per solver")
    click.echo("="*60)
    
    total_experiments = 0
    completed_experiments = 0
    
    for approach in approach_list:
        click.echo(f"\nTESTING {approach} APPROACH")
        click.echo("-" * 40)
        
        solvers = get_solvers_for_approach(approach)
        click.echo(f"Solvers to test: {', '.join(solvers)}")
        
        for n in range(4, max_n + 1, 2):
            click.echo(f"\nInstance: {n} teams")
            
            for solver in solvers:
                total_experiments += 1
                click.echo(f"  Running {approach}-{solver}...", nl=False)
                
                try:
                    start_time = time.time()
                    ctx = click.Context(solve)
                    ctx.invoke(solve, n=n, approach=approach, solver=solver, 
                             timeout=timeout, name=solver, output=None)
                    elapsed = time.time() - start_time
                    completed_experiments += 1
                    click.echo(f" OK ({elapsed:.1f}s)")
                    
                except Exception as e:
                    click.echo(f" Error: {str(e)[:50]}...")
    
    click.echo("\n" + "="*60)
    click.echo("BENCHMARK COMPLETE")
    click.echo("="*60)
    click.echo(f"Total experiments: {total_experiments}")
    click.echo(f"Completed successfully: {completed_experiments}")
    click.echo(f"Success rate: {100*completed_experiments/total_experiments:.1f}%")
    click.echo(f"\nResults saved in res/ directory by approach")
    click.echo("Use 'sts-solve validate-all --official' to validate all results")
    
    # Automatically run analysis after comprehensive benchmark
    if total_experiments > 10:  # Only for substantial runs
        click.echo("\n" + "="*60)
        click.echo("RUNNING POST-BENCHMARK ANALYSIS...")
        click.echo("="*60)
        
        try:
            from .utils.analytics import analyze_benchmark_results, print_comprehensive_report, validate_and_report_errors
            
            # Validate solutions first
            validate_and_report_errors(Path("res"))
            
            # Generate comprehensive analysis
            stats = analyze_benchmark_results(Path("res"))
            print_comprehensive_report(stats, Path("res"))
            
        except ImportError:
            click.echo("Analytics module not available")
        except Exception as e:
            click.echo(f"Error during analysis: {e}")


@cli.command()
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory")
@click.option("--format", type=click.Choice(["console", "json", "csv"]), default="console", help="Output format")
def analyze(results_dir: str, format: str):
    """Analyze benchmark results and generate comprehensive statistics"""
    
    from .utils.analytics import analyze_benchmark_results, print_comprehensive_report, validate_and_report_errors
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        click.echo(f"Error: Results directory {results_path} not found", err=True)
        return
    
    click.echo("Analyzing benchmark results...")
    
    # Validate all solutions
    validate_and_report_errors(results_path)
    
    # Generate comprehensive statistics
    stats = analyze_benchmark_results(results_path)
    
    if format == "console":
        print_comprehensive_report(stats, results_path)
    elif format == "json":
        # Export to JSON
        import json
        from dataclasses import asdict
        
        json_stats = {}
        for approach, stat in stats.items():
            json_stats[approach] = asdict(stat)
        
        output_file = results_path / "benchmark_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(json_stats, f, indent=2, default=str)
        click.echo(f"Analysis saved to {output_file}")
    
    elif format == "csv":
        # Export to CSV for spreadsheet analysis
        import csv
        
        output_file = results_path / "benchmark_summary.csv"
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Approach', 'Solver', 'Instance', 'Time', 'Optimal', 'Valid', 'Objective'])
            
            from .utils.analytics import load_all_results
            all_results = load_all_results(results_path)
            
            for result in all_results:
                writer.writerow([
                    result.approach, result.solver, result.instance, 
                    result.time, result.optimal, result.is_valid, result.obj
                ])
        
        click.echo(f"CSV data saved to {output_file}")


@cli.command() 
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory")
def validate_results(results_dir: str):
    """Validate all benchmark results and show detailed error report"""
    
    from .utils.analytics import validate_and_report_errors
    
    results_path = Path(results_dir)
    validate_and_report_errors(results_path)


@cli.command()
@click.argument("instance_size", type=int)
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory") 
def compare_instance(instance_size: int, results_dir: str):
    """Compare solver performance on a specific instance size"""
    
    from .utils.analytics import load_all_results
    
    results_path = Path(results_dir)
    all_results = load_all_results(results_path)
    
    # Filter results for specific instance
    instance_results = [r for r in all_results if r.instance == instance_size]
    
    if not instance_results:
        click.echo(f"No results found for instance size {instance_size}")
        return
    
    click.echo(f"PERFORMANCE COMPARISON: {instance_size} teams")
    click.echo("=" * 60)
    
    # Group by approach
    by_approach = {}
    for result in instance_results:
        if result.approach not in by_approach:
            by_approach[result.approach] = []
        by_approach[result.approach].append(result)
    
    for approach, results in by_approach.items():
        click.echo(f"\n{approach} Approach:")
        click.echo("-" * 30)
        
        # Sort by time (successful runs first, then by time)
        results.sort(key=lambda r: (not bool(r.sol), r.time))
        
        for result in results:
            status = "PASS" if result.is_valid else "FAIL" if result.sol else "TIMEOUT"
            opt_str = " (OPT)" if result.optimal else ""
            click.echo(f"  {status} {result.solver:<20} {result.time:>4}s{opt_str}")
    
    # Find overall winner
    valid_results = [r for r in instance_results if r.is_valid]
    if valid_results:
        fastest = min(valid_results, key=lambda r: r.time)
        click.echo(f"\nFASTEST: {fastest.approach}-{fastest.solver} ({fastest.time}s)")
        
        optimal_results = [r for r in valid_results if r.optimal]
        if optimal_results:
            fastest_optimal = min(optimal_results, key=lambda r: r.time)
            click.echo(f"FASTEST OPTIMAL: {fastest_optimal.approach}-{fastest_optimal.solver} ({fastest_optimal.time}s)")


@cli.command()
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory")
@click.option("--no-dry-run", is_flag=True, help="Actually clean files (default is dry-run mode)")
@click.option("--backup", is_flag=True, help="Create backup before cleaning")
def clean_results(results_dir: str, no_dry_run: bool, backup: bool):
    """Clean invalid solutions from result files"""
    
    from .utils.analytics import clean_invalid_solutions, backup_results
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        click.echo(f"Error: Results directory {results_path} not found", err=True)
        return
    
    # Create backup if requested
    if backup and not no_dry_run:
        backup_results(results_path)
        click.echo()
    
    # Clean invalid solutions
    dry_run = not no_dry_run
    clean_invalid_solutions(results_path, dry_run=dry_run)


@cli.command()
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory")
def backup_results_cmd(results_dir: str):
    """Create a timestamped backup of results directory"""
    
    from .utils.analytics import backup_results
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        click.echo(f"Error: Results directory {results_path} not found", err=True)
        return
    
    backup_path = backup_results(results_path)
    click.echo(f"Backup created at: {backup_path}")


@cli.command()
@click.option("--results-dir", "-r", type=click.Path(), default="res", help="Results directory")
@click.option("--solver", help="Specific solver to remove (e.g., 'GLOP')")
@click.option("--no-dry-run", is_flag=True, help="Actually remove solutions (default is dry-run)")
def remove_solver(results_dir: str, solver: Optional[str], no_dry_run: bool):
    """Remove all solutions from a specific solver"""
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        click.echo(f"Error: Results directory {results_path} not found", err=True)
        return
    
    if not solver:
        click.echo("Error: --solver parameter is required", err=True)
        return
    
    dry_run = not no_dry_run
    mode_str = "DRY RUN" if dry_run else "REMOVING"
    
    click.echo(f"SOLVER REMOVAL UTILITY ({mode_str})")
    click.echo("=" * 50)
    click.echo(f"Target solver: {solver}")
    click.echo(f"Directory: {results_path}")
    click.echo()
    
    files_modified = 0
    solutions_removed = 0
    
    for approach_dir in results_path.iterdir():
        if not approach_dir.is_dir():
            continue
            
        approach = approach_dir.name
        click.echo(f"Checking {approach} approach...")
        
        for result_file in approach_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                if solver in data:
                    click.echo(f"  📁 {result_file.relative_to(results_path)}: Found {solver}")
                    
                    if not dry_run:
                        # Remove the solver
                        del data[solver]
                        
                        if data:
                            # Write back the cleaned data
                            with open(result_file, 'w') as f:
                                json.dump(data, f, indent=2)
                        else:
                            # File would be empty - remove it
                            result_file.unlink()
                            click.echo(f"    Removed empty file")
                        
                    files_modified += 1
                    solutions_removed += 1
                    
            except (json.JSONDecodeError, FileNotFoundError):
                continue
    
    click.echo("\n" + "=" * 50)
    if dry_run:
        click.echo(f"Would remove {solutions_removed} instances of '{solver}' from {files_modified} files")
        click.echo("Use --no-dry-run to actually remove")
    else:
        click.echo(f"Removed {solutions_removed} instances of '{solver}' from {files_modified} files")
        click.echo("Cleaning completed!")


if __name__ == "__main__":
    cli()