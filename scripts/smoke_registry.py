from sts_solver.registry import registry
import sts_solver.mip.unified_bridge  # ensure registration
import sts_solver.smt.unified_bridge  # ensure registration
import sts_solver.cp.unified_bridge   # ensure registration
import sts_solver.sat.unified_bridge  # ensure registration

def main():
    print("Registered solvers:", registry.list_solvers())
    md = registry.get_metadata('SMT', 'baseline')
    print("SMT baseline metadata:", md)

if __name__ == "__main__":
    main()
