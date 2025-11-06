"""Benchmark commands (modular CLI)"""

import time
from typing import Optional, List
import click

from ..utils import get_solvers_for_approach
from ...config import get_config
from .solve import solve as solve_command


@click.command()
@click.argument("max_n", type=int)
@click.option("--approach", "-a", type=click.Choice(["CP", "SAT", "SMT", "MIP"]), default="CP")
@click.option("--solver", "-s", help="Specific solver to use")
@click.option("--timeout", "-t", type=int, default=None, help="Timeout in seconds (defaults from config)")
@click.option("--all-solvers", is_flag=True, help="Run all available solvers for the approach")
def benchmark(max_n: int, approach: str, solver: Optional[str], timeout: Optional[int], all_solvers: bool):
    """Run benchmark for instances from 4 to MAX_N teams."""

    cfg = get_config()
    effective_timeout = timeout if timeout is not None else cfg.benchmark_timeout

    if all_solvers:
        click.echo(f"Running comprehensive benchmark for {approach} approach up to {max_n} teams")
        solvers_to_test = get_solvers_for_approach(approach)
        click.echo(f"Testing solvers: {', '.join(solvers_to_test)}")
    else:
        click.echo(f"Running benchmark for {approach} approach up to {max_n} teams")
        if solver:
            click.echo(f"Using solver: {solver}")
        solvers_to_test = [solver] if solver else [None]

    for n in range(4, max_n + 1, 2):
        click.echo(f"\n{'='*50}")
        click.echo(f"INSTANCE: {n} teams")
        click.echo(f"{'='*50}")

        for test_solver in solvers_to_test:
            if test_solver:
                click.echo(f"\nTesting {approach} with {test_solver}...")
            else:
                click.echo(f"\nTesting {approach} with default solver...")

            try:
                start_time = time.time()
                # call click command's callback directly to avoid CLI context complexity
                cb = getattr(solve_command, 'callback', None)
                if cb is None:
                    raise RuntimeError('solve callback not available')
                cb(n=n, approach=approach, solver=test_solver, timeout=effective_timeout, output=None, optimization=False, name=test_solver)
                elapsed = time.time() - start_time
                click.echo(f"OK ({elapsed:.1f}s)")
            except Exception as e:
                click.echo(f"Error with {test_solver or 'default'}: {e}", err=True)


@click.command(name="comprehensive-benchmark")
@click.argument("max_n", type=int)
@click.option("--timeout", "-t", type=int, default=None, help="Timeout in seconds (defaults from config)")
@click.option("--approaches", help="Comma-separated list of approaches (CP,SAT,SMT,MIP)")
def comprehensive_benchmark(max_n: int, timeout: Optional[int], approaches: Optional[str]):
    """Run comprehensive benchmark across ALL approaches and solvers."""

    cfg = get_config()
    effective_timeout = timeout if timeout is not None else cfg.benchmark_timeout
    if approaches:
        approach_list = [a.strip().upper() for a in approaches.split(",")]
    else:
        approach_list = ["CP", "SAT", "SMT", "MIP"]

    click.echo("="*60)
    click.echo("COMPREHENSIVE STS SOLVER BENCHMARK")
    click.echo("="*60)
    click.echo(f"Approaches: {', '.join(approach_list)}")
    click.echo(f"Instance range: 4 to {max_n} teams")
    click.echo(f"Timeout: {effective_timeout} seconds per solver")
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
                    cb = getattr(solve_command, 'callback', None)
                    if cb is None:
                        raise RuntimeError('solve callback not available')
                    cb(n=n, approach=approach, solver=solver, timeout=effective_timeout, output=None, optimization=False, name=solver)
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
    if total_experiments:
        click.echo(f"Success rate: {100*completed_experiments/total_experiments:.1f}%")
    click.echo(f"\nResults saved in res/ directory by approach")
    click.echo("Use 'sts2 validate-results' to validate all results")
