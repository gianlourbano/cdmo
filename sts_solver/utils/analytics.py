"""
Benchmark analytics and reporting utilities

Provides comprehensive analysis of benchmark results including validation,
performance statistics, and comparative analysis.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import statistics
from collections import defaultdict

from .checker import check_solution_direct, CHECKER_AVAILABLE


@dataclass
class SolverResult:
    """Data class for individual solver results"""
    solver: str
    approach: str
    instance: int
    time: int
    optimal: bool
    obj: Optional[int]
    sol: List[List[List[int]]]
    is_valid: bool = False
    validation_message: str = ""


@dataclass
class BenchmarkStats:
    """Statistics for benchmark analysis"""
    total_runs: int = 0
    successful_runs: int = 0
    valid_solutions: int = 0
    invalid_solutions: int = 0
    optimal_solutions: int = 0
    timeout_runs: int = 0
    
    # Timing statistics
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    median_time: float = 0.0
    
    # Performance by instance size
    by_instance: Dict[int, Dict] = field(default_factory=dict)
    
    # Best performers
    fastest_solver: str = ""
    most_reliable: str = ""
    best_solver: str = ""


def analyze_benchmark_results(results_dir: Path = Path("res")) -> Dict[str, BenchmarkStats]:
    """
    Comprehensive analysis of all benchmark results
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        Dictionary mapping approach names to their statistics
    """
    
    all_results = load_all_results(results_dir)
    approach_stats = {}
    
    for approach in ["CP", "SAT", "SMT", "MIP"]:
        approach_results = [r for r in all_results if r.approach == approach]
        if approach_results:
            approach_stats[approach] = calculate_stats(approach_results)
    
    return approach_stats


def load_all_results(results_dir: Path) -> List[SolverResult]:
    """Load and validate all results from the results directory"""
    
    all_results = []
    
    for approach_dir in results_dir.iterdir():
        if not approach_dir.is_dir():
            continue
            
        approach = approach_dir.name
        
        for result_file in approach_dir.glob("*.json"):
            try:
                instance = int(result_file.stem)
                
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                for solver, result in data.items():
                    solver_result = SolverResult(
                        solver=solver,
                        approach=approach,
                        instance=instance,
                        time=result.get("time", 300),
                        optimal=result.get("optimal", False),
                        obj=result.get("obj"),
                        sol=result.get("sol", [])
                    )
                    
                    # Check for known problematic solvers first
                    if "GLOP" in solver.upper():
                        solver_result.is_valid = False
                        solver_result.validation_message = "GLOP is LP-only solver, produces invalid discrete solutions"
                    # Validate solution if checker is available
                    elif CHECKER_AVAILABLE and solver_result.sol:
                        is_valid, message = check_solution_direct(
                            solver_result.sol, 
                            solver_result.obj, 
                            solver_result.time, 
                            solver_result.optimal
                        )
                        solver_result.is_valid = is_valid
                        solver_result.validation_message = message
                    else:
                        solver_result.is_valid = bool(solver_result.sol)  # Basic check
                        solver_result.validation_message = "No checker available"
                    
                    all_results.append(solver_result)
                    
            except (ValueError, json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {result_file}: {e}")
                continue
    
    return all_results


def calculate_stats(results: List[SolverResult]) -> BenchmarkStats:
    """Calculate comprehensive statistics for a set of results"""
    
    stats = BenchmarkStats()
    
    if not results:
        return stats
    
    # Basic counts
    stats.total_runs = len(results)
    stats.successful_runs = len([r for r in results if r.sol])
    stats.valid_solutions = len([r for r in results if r.is_valid])
    stats.invalid_solutions = len([r for r in results if r.sol and not r.is_valid])
    stats.optimal_solutions = len([r for r in results if r.optimal])
    stats.timeout_runs = len([r for r in results if r.time >= 300])
    
    # Timing statistics (only for successful runs)
    successful_results = [r for r in results if r.sol]
    if successful_results:
        times = [r.time for r in successful_results]
        stats.total_time = sum(times)
        stats.min_time = min(times)
        stats.max_time = max(times)
        stats.avg_time = statistics.mean(times)
        stats.median_time = statistics.median(times)
    
    # Statistics by instance size
    by_instance = defaultdict(lambda: {
        'attempts': 0, 'successes': 0, 'valid': 0, 
        'optimal': 0, 'times': [], 'solvers': []
    })
    
    for result in results:
        n = result.instance
        by_instance[n]['attempts'] += 1
        if result.sol:
            by_instance[n]['successes'] += 1
            by_instance[n]['times'].append(result.time)
            by_instance[n]['solvers'].append(result.solver)
        if result.is_valid:
            by_instance[n]['valid'] += 1
        if result.optimal:
            by_instance[n]['optimal'] += 1
    
    # Calculate derived stats for each instance
    for n, data in by_instance.items():
        data['success_rate'] = data['successes'] / data['attempts'] if data['attempts'] > 0 else 0
        data['avg_time'] = statistics.mean(data['times']) if data['times'] else 0
        data['fastest_time'] = min(data['times']) if data['times'] else 0
        data['fastest_solver'] = data['solvers'][data['times'].index(data['fastest_time'])] if data['times'] else ""
    
    stats.by_instance = dict(by_instance)
    
    # Find best performers
    solver_performance = defaultdict(lambda: {'successes': 0, 'total_time': 0, 'attempts': 0, 'valid': 0})
    
    for result in results:
        perf = solver_performance[result.solver]
        perf['attempts'] += 1
        if result.sol:
            perf['successes'] += 1
            perf['total_time'] += result.time
        if result.is_valid:
            perf['valid'] += 1
    
    # Calculate performance metrics
    for solver, perf in solver_performance.items():
        perf['success_rate'] = perf['successes'] / perf['attempts'] if perf['attempts'] > 0 else 0
        perf['avg_time'] = perf['total_time'] / perf['successes'] if perf['successes'] > 0 else float('inf')
        perf['reliability'] = perf['valid'] / perf['attempts'] if perf['attempts'] > 0 else 0
    
    # Find best solvers
    if solver_performance:
        # Fastest average solver (among those with >50% success rate)
        fast_candidates = {s: p for s, p in solver_performance.items() if p['success_rate'] >= 0.5}
        if fast_candidates:
            stats.fastest_solver = min(fast_candidates, key=lambda s: fast_candidates[s]['avg_time'])
        
        # Most reliable solver
        stats.most_reliable = max(solver_performance, key=lambda s: solver_performance[s]['reliability'])
        
        # Best overall solver (weighted combination of speed and reliability)
        def overall_score(solver):
            perf = solver_performance[solver]
            if perf['success_rate'] == 0 or perf['reliability'] == 0:
                return 0
            # Higher is better: reliability / log(avg_time + 1)
            import math
            log_time = math.log(perf['avg_time'] + 1)
            if log_time == 0:
                return perf['reliability']
            return perf['reliability'] / log_time
        
        stats.best_solver = max(solver_performance, key=overall_score)
    
    return stats


def print_comprehensive_report(stats: Dict[str, BenchmarkStats], results_dir: Path = Path("res")):
    """Print a comprehensive benchmark analysis report"""
    
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK ANALYSIS REPORT")
    print("=" * 80)
    print(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results Directory: {results_dir.absolute()}")
    print()
    
    # Overall summary
    total_runs = sum(s.total_runs for s in stats.values())
    total_valid = sum(s.valid_solutions for s in stats.values())
    total_successful = sum(s.successful_runs for s in stats.values())
    
    print("OVERALL SUMMARY")
    print("-" * 40)
    print(f"Total Experiments: {total_runs}")
    print(f"Successful Runs: {total_successful} ({100*total_successful/total_runs:.1f}%)")
    print(f"Valid Solutions: {total_valid} ({100*total_valid/total_runs:.1f}%)")
    print()
    
    # Approach-by-approach analysis
    for approach, approach_stats in stats.items():
        print(f"{approach} APPROACH ANALYSIS")
        print("-" * 50)
        print(f"Total Runs: {approach_stats.total_runs}")
        print(f"Success Rate: {100*approach_stats.successful_runs/approach_stats.total_runs:.1f}%")
        print(f"Valid Solutions: {approach_stats.valid_solutions}/{approach_stats.total_runs}")
        
        if approach_stats.invalid_solutions > 0:
            print(f"Invalid Solutions: {approach_stats.invalid_solutions}")
        
        if approach_stats.successful_runs > 0:
            print(f"Average Time: {approach_stats.avg_time:.1f}s")
            print(f"Fastest Time: {approach_stats.min_time}s")
            print(f"Slowest Time: {approach_stats.max_time}s")
            print(f"Fastest Solver: {approach_stats.fastest_solver}")
            print(f"Most Reliable: {approach_stats.most_reliable}")
            print(f"Best Overall: {approach_stats.best_solver}")
        
        # Instance size breakdown
        print("\nPerformance by Instance Size:")
        for n in sorted(approach_stats.by_instance.keys()):
            data = approach_stats.by_instance[n]
            print(f"  n={n}: {data['successes']}/{data['attempts']} solved "
                  f"({100*data['success_rate']:.0f}%) - "
                  f"avg: {data['avg_time']:.1f}s, "
                  f"fastest: {data['fastest_time']}s ({data['fastest_solver']})")
        
        print()
    
    # Cross-approach comparison
    print("CROSS-APPROACH COMPARISON")
    print("-" * 40)
    
    # Find hardest instances
    all_instances = set()
    for s in stats.values():
        all_instances.update(s.by_instance.keys())
    
    if all_instances:
        print("Most Challenging Instances:")
        instance_difficulty = {}
        for n in all_instances:
            total_attempts = sum(s.by_instance.get(n, {}).get('attempts', 0) for s in stats.values())
            total_successes = sum(s.by_instance.get(n, {}).get('successes', 0) for s in stats.values())
            if total_attempts > 0:
                success_rate = total_successes / total_attempts
                instance_difficulty[n] = 1 - success_rate  # Higher = more difficult
        
        hardest = sorted(instance_difficulty.items(), key=lambda x: x[1], reverse=True)[:5]
        for n, difficulty in hardest:
            success_rate = 1 - difficulty
            print(f"  n={n}: {100*success_rate:.0f}% success rate")
    
    # Best solvers across all approaches
    print("\nTOP PERFORMERS")
    print("-" * 30)
    all_results = []
    for approach_stats in stats.values():
        # We'd need to track solver performance across approaches
        # For now, show per-approach winners
        if approach_stats.fastest_solver:
            print(f"Fastest {approach_stats.fastest_solver}")
        if approach_stats.most_reliable:
            print(f"Most Reliable {approach_stats.most_reliable}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Generate recommendations based on analysis
    for approach, approach_stats in stats.items():
        if approach_stats.successful_runs > 0:
            print(f"\n{approach} Approach:")
            if approach_stats.best_solver:
                print(f"  • Best overall solver: {approach_stats.best_solver}")
            
            # Instance-specific recommendations
            easy_instances = [n for n, data in approach_stats.by_instance.items() 
                            if data['success_rate'] >= 0.8]
            hard_instances = [n for n, data in approach_stats.by_instance.items() 
                            if data['success_rate'] < 0.5]
            
            if easy_instances:
                print(f"  • Reliable for instances: n ≤ {max(easy_instances)}")
            if hard_instances:
                print(f"  • Challenging for instances: n ≥ {min(hard_instances)}")
    
    print("\n" + "=" * 80)


def clean_invalid_solutions(results_dir: Path = Path("res"), dry_run: bool = True) -> Dict[str, int]:
    """
    Clean invalid solutions from result files
    
    Args:
        results_dir: Directory containing results
        dry_run: If True, only report what would be cleaned without modifying files
        
    Returns:
        Dictionary with cleaning statistics
    """
    
    if not CHECKER_AVAILABLE and not dry_run:
        print("Cannot clean without official solution checker - run in dry-run mode only")
        return {}
    
    print("SOLUTION CLEANING UTILITY")
    print("=" * 50)
    print(f"Mode: {'DRY RUN (preview only)' if dry_run else 'CLEANING (will modify files)'}")
    print(f"Directory: {results_dir}")
    print()
    
    all_results = load_all_results(results_dir)
    invalid_results = [r for r in all_results if r.sol and not r.is_valid]
    
    cleaning_stats = {
        'total_files_checked': 0,
        'files_modified': 0,
        'invalid_solutions_removed': 0,
        'solvers_removed': set()
    }
    
    if not invalid_results:
        print("No invalid solutions found - nothing to clean!")
        return cleaning_stats
    
    print(f"Found {len(invalid_results)} invalid solutions to clean:")
    
    # Group by file for cleaning
    by_file = {}
    for result in invalid_results:
        file_key = (result.approach, result.instance)
        if file_key not in by_file:
            by_file[file_key] = []
        by_file[file_key].append(result)
    
    for (approach, instance), invalid_solutions in by_file.items():
        file_path = results_dir / approach / f"{instance}.json"
        cleaning_stats['total_files_checked'] += 1
        
        print(f"\n{file_path.relative_to(results_dir)}")
        
        # Load current file data
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"  Could not read file - skipping")
            continue
        
        # Identify solvers to remove
        solvers_to_remove = [r.solver for r in invalid_solutions]
        print(f"  Invalid solvers to remove: {', '.join(solvers_to_remove)}")
        
        for solver in solvers_to_remove:
            cleaning_stats['solvers_removed'].add(solver)
        
        # Show what would be kept
        remaining_solvers = [s for s in data.keys() if s not in solvers_to_remove]
        if remaining_solvers:
            print(f"  Valid solvers to keep: {', '.join(remaining_solvers)}")
        else:
            print(f"  Warning: No valid solutions left - file would become empty!")
        
        if not dry_run:
            # Actually clean the file
            cleaned_data = {k: v for k, v in data.items() if k not in solvers_to_remove}
            
            if cleaned_data:
                # Write cleaned data back
                with open(file_path, 'w') as f:
                    json.dump(cleaned_data, f, indent=2)
                cleaning_stats['files_modified'] += 1
                cleaning_stats['invalid_solutions_removed'] += len(solvers_to_remove)
                print(f"  Cleaned - removed {len(solvers_to_remove)} invalid solver(s)")
            else:
                # File would be empty - remove it entirely
                file_path.unlink()
                cleaning_stats['files_modified'] += 1 
                cleaning_stats['invalid_solutions_removed'] += len(solvers_to_remove)
                print(f"  File removed - no valid solutions remaining")
    
    # Summary
    print("\n" + "=" * 50)
    print("CLEANING SUMMARY")
    print("=" * 50)
    print(f"Files checked: {cleaning_stats['total_files_checked']}")
    
    if dry_run:
        print(f"Files that would be modified: {len(by_file)}")
        print(f"Invalid solutions that would be removed: {len(invalid_results)}")
        print(f"Problem solvers identified: {', '.join(sorted(cleaning_stats['solvers_removed']))}")
        print("\nRun with --no-dry-run to actually clean the files")
    else:
        print(f"Files actually modified: {cleaning_stats['files_modified']}")
        print(f"Invalid solutions removed: {cleaning_stats['invalid_solutions_removed']}")
        print(f"Problem solvers cleaned: {', '.join(sorted(cleaning_stats['solvers_removed']))}")
        print("\nCleaning completed!")
    
    return cleaning_stats


def backup_results(results_dir: Path = Path("res")) -> Path:
    """
    Create a backup of results before cleaning
    
    Args:
        results_dir: Directory to backup
        
    Returns:
        Path to backup directory
    """
    import shutil
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = results_dir.parent / f"res_backup_{timestamp}"
    
    print(f"Creating backup: {backup_dir}")
    shutil.copytree(results_dir, backup_dir)
    print(f"Backup created successfully")
    
    return backup_dir


def validate_and_report_errors(results_dir: Path = Path("res")):
    """Validate all solutions and report detailed error information"""
    
    if not CHECKER_AVAILABLE:
        print("Official solution checker not available - cannot validate solutions")
        return
    
    print("SOLUTION VALIDATION REPORT")
    print("=" * 50)
    
    all_results = load_all_results(results_dir)
    invalid_results = [r for r in all_results if r.sol and not r.is_valid]
    
    if not invalid_results:
        print("All solutions are VALID!")
        return
    
    print(f"Found {len(invalid_results)} INVALID solutions:")
    print()
    
    # Group by error type
    error_groups = defaultdict(list)
    for result in invalid_results:
        error_groups[result.validation_message].append(result)
    
    for error_msg, results in error_groups.items():
        print(f"Error: {error_msg}")
        print(f"   Affected runs: {len(results)}")
        
        # Show affected solvers and instances
        solvers = list(set(r.solver for r in results))
        instances = sorted(set(r.instance for r in results))
        
        print(f"   Solvers: {', '.join(solvers)}")
        print(f"   Instances: {instances}")
        print()