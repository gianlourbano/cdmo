"""Benchmark commands (modular CLI) with optional optimization support.

Adds `--optimization/-O` flag so that all benchmark styles (single approach
and comprehensive) can invoke solvers in optimization mode when they
support it. Objective values (if any) are printed after each run.
"""

import json
import time
from pathlib import Path
from typing import Optional, List
import click

from ..utils import get_solvers_for_approach
from ...config import get_config
from ...registry import registry
from .solve import solve as solve_command


@click.command()
@click.argument("max_n", type=int)
@click.option("--approach", "-a", type=click.Choice(["CP", "SAT", "SMT", "MIP"]), default="CP")
@click.option("--model", "-m", help="Specific model (formulation) to use")
@click.option("--solver", "-s", help="Backend only for CP/MIP (e.g., gecode, CBC, SCIP)")
@click.option("--timeout", "-t", type=int, default=300, help="Timeout in seconds (defaults from config)")
@click.option("--all-solvers", is_flag=True, help="Run all available models for the approach")
@click.option("--optimization", "-O", is_flag=True, help="Enable optimization mode (only models supporting it)")
def benchmark(max_n: int, approach: str, model: Optional[str], solver: Optional[str], timeout: Optional[int], all_solvers: bool, optimization: bool):
    """Run benchmark for instances from 4 to MAX_N teams.

    When `--optimization` is set, only solvers declaring optimization support
    are exercised and their objective values (if present) are reported.
    """

    cfg = get_config()
    effective_timeout = timeout if timeout is not None else cfg.benchmark_timeout

    if all_solvers:
        click.echo(f"Running comprehensive benchmark for {approach} approach up to {max_n} teams")
        solvers_to_test = get_solvers_for_approach(approach)
        # Filter for optimization support if requested
        if optimization:
            filtered: List[str] = []
            for s in solvers_to_test:
                md = registry.get_metadata(approach, s)
                if md and md.supports_optimization:
                    filtered.append(s)
            solvers_to_test = filtered
        click.echo(f"Testing solvers: {', '.join(solvers_to_test)}")
    else:
        click.echo(f"Running benchmark for {approach} approach up to {max_n} teams")
        if model:
            click.echo(f"Using model: {model}")
        if solver:
            if approach in {"SAT", "SMT"}:
                click.echo(f"Note: '{approach}' ignores --solver backend", err=True)
            else:
                click.echo(f"Backend: {solver}")
        # Single model or default selection; default will be chosen via registry taking optimization flag into account
        solvers_to_test = [model] if model else [None]

    for n in range(4, max_n + 1, 2):
        click.echo(f"\n{'='*50}")
        click.echo(f"INSTANCE: {n} teams")
        click.echo(f"{'='*50}")

        for test_model in solvers_to_test:
            chosen_model_name: Optional[str] = test_model
            if test_model:
                click.echo(f"\nTesting {approach} with {test_model}...")
            else:
                chosen_model_name = registry.find_best_solver(approach, n, optimization)
                click.echo(f"\nTesting {approach} with default model ({chosen_model_name})...")

            try:
                start_time = time.time()
                cb = getattr(solve_command, 'callback', None)
                if cb is None:
                    raise RuntimeError('solve callback not available')
                # Use opt- prefix in saved key when either optimization is enabled
                # OR the model name starts with "opt_" (MiniZinc optimization models)
                result_name = None
                if chosen_model_name:
                    is_opt_model = chosen_model_name.lower().startswith("opt_")
                    result_name = f"opt-{chosen_model_name}" if (optimization or is_opt_model) else chosen_model_name
                cb(n=n, approach=approach, model=chosen_model_name, solver=solver, timeout=effective_timeout, output=None, optimization=optimization, name=result_name)
                elapsed = time.time() - start_time

                # Read objective if optimization enabled
                if optimization and chosen_model_name:
                    cfg = get_config()
                    results_path = cfg.results_dir / approach / f"{n}.json"
                    if results_path.exists():
                        try:
                            with open(results_path) as rf:
                                data = json.load(rf)
                            # Look under the exact saved key used earlier
                            key = result_name or f"opt-{chosen_model_name}"
                            entry = data.get(key)
                            obj_val = entry.get("obj") if entry else None
                            if obj_val is not None:
                                click.echo(f"OK ({elapsed:.1f}s) objective={obj_val}")
                            else:
                                click.echo(f"OK ({elapsed:.1f}s) objective=NA")
                        except Exception:
                            click.echo(f"OK ({elapsed:.1f}s) objective=ERROR")
                    else:
                        click.echo(f"OK ({elapsed:.1f}s) objective=FILE_MISSING")
                else:
                    click.echo(f"OK ({elapsed:.1f}s)")
            except Exception as e:
                label = test_model or 'default'
                click.echo(f"Error with {label}: {e}", err=True)


@click.command(name="comprehensive-benchmark")
@click.argument("max_n", type=int)
@click.option("--timeout", "-t", type=int, default=None, help="Timeout in seconds (defaults from config)")
@click.option("--approaches", help="Comma-separated list of approaches (CP,SAT,SMT,MIP)")
@click.option("--optimization", "-O", is_flag=True, help="Enable optimization mode when supported")
def comprehensive_benchmark(max_n: int, timeout: Optional[int], approaches: Optional[str], optimization: bool):
    """Run comprehensive benchmark across ALL approaches and solvers.

    With `--optimization`, only solvers advertising optimization support are
    exercised and their objective values are printed.
    """

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
        if optimization:
            filtered2: List[str] = []
            for s in solvers:
                md = registry.get_metadata(approach, s)
                if md and md.supports_optimization:
                    filtered2.append(s)
            solvers = filtered2
        click.echo(f"Solvers to test: {', '.join(solvers) if solvers else '(none)'}")

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
                    # Save optimized runs under opt-<model> to avoid overwrites
                    is_opt_model = solver.lower().startswith("opt_")
                    result_name = f"opt-{solver}" if (optimization or is_opt_model) else solver
                    cb(n=n, approach=approach, model=solver, solver=None, timeout=effective_timeout, output=None, optimization=optimization, name=result_name)
                    elapsed = time.time() - start_time
                    completed_experiments += 1

                    if optimization:
                        cfg = get_config()
                        results_path = cfg.results_dir / approach / f"{n}.json"
                        obj_txt = "obj=NA"
                        if results_path.exists():
                            try:
                                with open(results_path) as rf:
                                    data = json.load(rf)
                                entry = data.get(f"opt-{solver}")
                                if entry and entry.get("obj") is not None:
                                    obj_txt = f"obj={entry.get('obj')}"
                            except Exception:
                                obj_txt = "obj=ERR"
                        click.echo(f" OK ({elapsed:.1f}s, {obj_txt})")
                    else:
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
