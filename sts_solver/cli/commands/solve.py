"""Solve command implemented against the unified registry"""

import time
from pathlib import Path
from typing import Optional

import click

from ...registry import registry
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

    output_dir = Path(output) if output else Path("res") / approach

    click.echo(f"Solving STS instance with {n} teams using {approach} approach")
    if solver:
        click.echo(f"Using solver: {solver}")

    chosen = solver or registry.find_best_solver(approach, n, optimization)
    if not chosen:
        click.echo(f"Error: No suitable solver found for {approach}", err=True)
        raise SystemExit(1)

    solver_cls = registry.get_solver(approach, chosen)
    if not solver_cls:
        click.echo(f"Error: Solver '{chosen}' not found", err=True)
        raise SystemExit(1)

    start_time = time.time()
    instance = solver_cls(n, timeout, optimization)
    result = instance.solve()

    result_name = name or chosen
    save_results(n, approach, {result_name: result}, output_dir)

    click.echo(f"Solution completed in {time.time() - start_time:.2f}s")
    click.echo(f"Results saved to {output_dir / f'{n}.json'}")
