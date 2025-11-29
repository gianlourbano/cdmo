"""Solve command implemented against the unified registry"""

import time
from pathlib import Path
from typing import Optional

import click

from ...registry import registry
from ...config import get_config
from ...utils.error_handling import handle_solver_errors
from ...utils.solution_format import save_results
from ..utils import validate_instance_size


@click.command()
@click.argument("n", type=int)
@click.argument("approach", type=click.Choice(["CP", "SAT", "SMT", "MIP"]))
@click.option("--model", "-m", help="Model (formulation) name registered for the approach")
@click.option("--solver", "-s", help="Backend only (CP: gecode/chuffed, MIP: CBC/SCIP/...) — not a model")
@click.option("--timeout", "-t", type=int, default=None, help="Timeout in seconds (defaults from config)")
@click.option("--output", "-o", type=click.Path(), help="Output directory (defaults from config)")
@click.option("--optimization", is_flag=True, help="Enable optimization objective if supported")
@click.option("--name", help="Custom name for results JSON key")
def solve(
    n: int,
    approach: str,
    model: Optional[str],
    solver: Optional[str],
    timeout: int,
    output: Optional[str],
    optimization: bool,
    name: Optional[str],
):
    """Solve STS instance with N teams using specified approach (modular CLI)."""

    # Ensure registration side-effects for approaches we support now
    import sts_solver.mip.unified_bridge  # noqa: F401
    import sts_solver.smt.unified_bridge  # noqa: F401
    import sts_solver.cp.unified_bridge   # noqa: F401
    import sts_solver.sat.unified_bridge  # noqa: F401

    try:
        validate_instance_size(n)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    cfg = get_config()
    effective_timeout = timeout if timeout is not None else cfg.default_timeout
    output_dir = Path(output) if output else cfg.results_dir / approach

    click.echo(f"Solving STS instance with {n} teams using {approach} approach")
    if model:
        click.echo(f"Model: {model}")
    if solver:
        click.echo(f"Backend: {solver}")

    # Strict separation: --model selects formulation; --solver selects backend only for CP/MIP.
    # No legacy hyphen parsing; no cross-interpretation.
    backend: Optional[str] = None
    raw_model = model

    # Enforce solver usage per approach
    if solver and approach in {"SAT", "SMT"}:
        click.echo(f"Error: '{approach}' does not accept a backend via --solver", err=True)
        raise SystemExit(1)
    if solver and approach in {"MIP", "CP"}:
        backend = solver

    chosen_model = raw_model
    if not chosen_model:
        click.echo(f"Error: No suitable model found for {approach}. Use --model to choose.", err=True)
        raise SystemExit(1)

    solver_cls = registry.get_solver(approach, chosen_model)
    if not solver_cls:
        click.echo(f"Error: Model '{chosen_model}' not found", err=True)
        raise SystemExit(1)

    start_time = time.time()
    # Pass search strategy dynamically by setting attribute (used in CP path)
    instance = solver_cls(n, effective_timeout, optimization)
    # Attach backend if supported.
    if backend:
        setattr(instance, "backend", backend)  # type: ignore[attr-defined]

    with handle_solver_errors(rethrow=False):
        result = instance.solve()

    # Default naming: prefix with "opt-" when either optimization is enabled
    # OR the chosen model name starts with "opt_" (optimization MiniZinc models),
    # to avoid overwriting non-optimized runs. If a custom --name is provided, respect it.
    is_opt_model = isinstance(chosen_model, str) and chosen_model.lower().startswith("opt_")
    result_name = name or (f"opt-{chosen_model}" if (optimization or is_opt_model) else chosen_model)
    if backend:
        result_name = f"{result_name}-{backend}"
    save_results(n, approach, {result_name: result}, output_dir)

    click.echo(f"Solution completed in {time.time() - start_time:.2f}s")
    click.echo(f"Results saved to {output_dir / f'{n}.json'}")
    if backend:
        click.echo(f"Backend recorded: {backend}")
