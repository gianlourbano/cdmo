"""List solvers command (modular CLI)"""

import click

from ...registry import registry


@click.command(name="list-solvers")
@click.option("--solvers", is_flag=True, help="List available solvers")
def list_solvers(solvers: bool):
    """List available solvers and formulations."""
    # Ensure registrations
    import sts_solver.mip.unified_bridge  # noqa: F401
    import sts_solver.smt.unified_bridge  # noqa: F401
    import sts_solver.cp.unified_bridge   # noqa: F401
    import sts_solver.sat.unified_bridge  # noqa: F401

    if not solvers:
        click.echo("Pass --solvers to list registered solvers")
        return

    all_solvers = registry.list_solvers()
    for approach, names in sorted(all_solvers.items()):
        click.echo(f"=== {approach} Solvers ===")
        if names:
            for name in sorted(names):
                md = registry.get_metadata(approach, name)
                desc = f" - {md.description}" if md and md.description else ""
                click.echo(f"  - {name}{desc}")
        else:
            click.echo("  (none)")
