"""List models/formulations command (modular CLI)

Renamed from list-solvers to list-models. Shows registered model names
per approach from the unified registry. For CP, entries correspond to
MiniZinc backend wrappers.
"""

import click

from ...registry import registry


@click.command(name="list-models")
@click.option("--approach", "-a", help="Filter by approach (CP, SAT, SMT, MIP)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed model information")
def list_models(approach: str, verbose: bool):
    """List available models (formulations) registered per approach."""
    # Ensure registrations
    import sts_solver.mip.unified_bridge  # noqa: F401
    import sts_solver.smt.unified_bridge  # noqa: F401
    import sts_solver.cp.unified_bridge   # noqa: F401
    import sts_solver.sat.unified_bridge  # noqa: F401

    all_solvers = registry.list_solvers(approach=approach.upper() if approach else None)
    
    for app, names in sorted(all_solvers.items()):
        click.echo(f"=== {app} Models ===")
        if names:
            for name in sorted(names):
                md = registry.get_metadata(app, name)
                if verbose and md:
                    click.echo(f"  - {name}")
                    click.echo(f"      Description: {md.description or 'N/A'}")
                    click.echo(f"      Version: {md.version}")
                    click.echo(f"      Optimization: {'Yes' if md.supports_optimization else 'No'}")
                    if md.max_recommended_size:
                        click.echo(f"      Max recommended size: {md.max_recommended_size}")
                else:
                    desc = f" - {md.description}" if md and md.description else ""
                    click.echo(f"  - {name}{desc}")
        else:
            click.echo("  (none)")
        click.echo()
