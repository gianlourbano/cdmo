import cvc5

from ..utils.solution_format import STSSolution
from .registry import register_smt_solver
from typing import Optional

@register_smt_solver("presolve_cvc5")
def presolve_cvc5(
        n: int,
        solver_name: Optional[str] = None,
        timeout: int = 300,
        optimization: bool = False
) -> STSSolution:
    

    src_folder = "source/SMT"

    solver = cvc5.Solver()
    solver.setOption("incremental", "true")
    solver.setOption("produce-models", "true")

    parser = cvc5.InputParser(solver)

    input_file = f"{src_folder}/from_z3__{n}.smt2"

    file_contents = open(input_file, "r").read()
    parser.setStringInput(cvc5.InputLanguage.SMT_LIB_2_6, file_contents, "input.smt2")

    sm = parser.getSymbolManager()

    model = None

    while True:
        cmd = parser.nextCommand()
        if cmd.isNull():
            break
        #print(f"Executing command {cmd}:")
        # invoke the command on the solver and the symbol manager, print the result
        
        if cmd.getCommandName() == "get-model":
            model = cmd.invoke(solver, sm)
        else:
            cmd.invoke(solver, sm)

    # now, check sat with the solver
    #r = solver.checkSat()
    #print(f"result: {r}")

        # need to parse the model into our format.
        # variables are of the form p_w_i_j where w is week, i,j are teams and the value is the period assigned
        # we need to build a list of periods, each period is a list of weeks, each week is a list of games (two teams)
        # so the final format is List[List[List[int]]]
        if model is not None:
            # model is a string of the form:
            # (define-fun p_1_1_2 () Int 1)
            # (define-fun p_1_1_3 () Int 2)
            # ...
            # we can parse it line by line

            lines = str(model).splitlines()
            periods = []  # periods[period][week] = [team1, team2]
            for line in lines:
                if line.startswith("(define-fun p_"):
                    parts = line.split()
                    var_name = parts[1]
                    value = int(parts[-1].strip(")"))
                    _, w, i, j = var_name.split("_")
                    w = int(w) - 1  # zero-based index
                    i = int(i)
                    j = int(j)
                    period = value - 1  # zero-based index
                    # ensure periods has enough periods
                    while len(periods) <= period:
                        periods.append([[] for _ in range(n - 1)])  # n-1 weeks per period
                    periods[period][w] = [i, j]  # assign teams to the week in this period
            return STSSolution(time=0, optimal=True, obj=None, sol=periods)
        

    return STSSolution(time=0, optimal=False, obj=None, sol=[])
