import math
from itertools import combinations
from typing import List, Any, Dict, Tuple
from z3 import Solver, Bool, Not, Or, And, Implies, BoolVal

# --- Problem Decomposition ---

def circle_matchings(n: int) -> Dict[int, List[Tuple[int, int]]]:
    """Generates canonical weekly matchups using the Circle Method."""
    if n % 2 != 0:
        raise ValueError("Circle method requires an even number of teams.")
    teams = list(range(1, n + 1))
    schedule = {}
    for w in range(1, n):
        week_matches = []
        pivot = teams[-1]
        week_matches.append(tuple(sorted((pivot, teams[w-1]))))
        for i in range(n // 2 - 1):
            team1 = teams[(w + i) % (n - 1)]
            team2 = teams[(w - i - 2 + (n - 1)) % (n - 1)]
            week_matches.append(tuple(sorted((team1, team2))))
        schedule[w] = week_matches
    return schedule

# --- Pairwise Cardinality Constraints ---

def at_most_1_pairwise(solver: Solver, literals: List[Any]):
    """Enforces Sum(literals) <= 1 using pairwise encoding (O(n^2) clauses)."""
    for l1, l2 in combinations(literals, 2):
        solver.add(Or(Not(l1), Not(l2)))

def at_least_1(solver: Solver, literals: List[Any]):
    """Enforces Sum(literals) >= 1 (simple OR clause)."""
    if not literals:
        solver.add(BoolVal(False))
        return
    solver.add(Or(*literals))

def exactly_1_pairwise(solver: Solver, literals: List[Any]):
    """Enforces Sum(literals) == 1 using pairwise encoding."""
    at_most_1_pairwise(solver, literals)
    at_least_1(solver, literals)

def at_most_2_pairwise(solver: Solver, literals: List[Any]):
    """Enforces Sum(literals) <= 2 using pairwise encoding (O(n^3) clauses)."""
    for l1, l2, l3 in combinations(literals, 3):
        solver.add(Or(Not(l1), Not(l2), Not(l3)))

# --- Sequential Counter Cardinality Constraints ---

def sequential_at_most_k(solver: Solver, literals: List[Any], k: int, prefix: str = ""):
    """Enforces Sum(literals) <= k using Sequential Counter."""
    n = len(literals)
    if k >= n: return
    if k < 0:
        for l in literals: solver.add(Not(l))
        return

    # s[i][j] means "sum of first i+1 literals is >= j"
    # j goes from 1 to k
    s = [[Bool(f"{prefix}_s_{i}_{j}") for j in range(k + 1)] for i in range(n)]
    
    # Base case: i=0
    solver.add(s[0][1] == literals[0])
    for j in range(2, k + 1):
        solver.add(Not(s[0][j]))
        
    # Induction
    for i in range(1, n):
        # j=1: sum >= 1 if prev >= 1 OR current is True
        solver.add(s[i][1] == Or(s[i-1][1], literals[i]))
        
        for j in range(2, k + 1):
            # sum >= j if prev >= j OR (prev >= j-1 AND current is True)
            solver.add(s[i][j] == Or(s[i-1][j], And(s[i-1][j-1], literals[i])))
            
        # Constraint: At Most k
        # Cannot have sum >= k+1.
        # This occurs if we already had k (s[i-1][k]) AND we add one more (literals[i])
        solver.add(Not(And(s[i-1][k], literals[i])))

def sequential_at_least_k(solver: Solver, literals: List[Any], k: int, prefix: str = ""):
    """Enforces Sum(literals) >= k using Sequential Counter."""
    n = len(literals)
    if k <= 0: return
    if n < k:
        solver.add(BoolVal(False))
        return
        
    s = [[Bool(f"{prefix}_al_{i}_{j}") for j in range(k + 1)] for i in range(n)]
    
    # Base case: i=0
    solver.add(s[0][1] == literals[0])
    for j in range(2, k + 1):
        solver.add(Not(s[0][j]))
        
    # Induction
    for i in range(1, n):
        solver.add(s[i][1] == Or(s[i-1][1], literals[i]))
        for j in range(2, k + 1):
            solver.add(s[i][j] == Or(s[i-1][j], And(s[i-1][j-1], literals[i])))
            
    # Constraint: At Least k
    # The final sum must be >= k
    solver.add(s[n-1][k])

def sequential_exactly_2(solver: Solver, literals: List[Any], prefix: str = ""):
    """
    Optimized Sequential Counter specifically for Sum(literals) == 2.
    Uses fewer auxiliary variables than the generic k version.
    """
    n = len(literals)
    if n < 2:
        solver.add(BoolVal(False))
        return

    s1 = [Bool(f"{prefix}_sq2_ge1_{i}") for i in range(n)]
    s2 = [Bool(f"{prefix}_sq2_ge2_{i}") for i in range(n)]

    # Base case i=0
    solver.add(s1[0] == literals[0])
    solver.add(Not(s2[0])) 

    for i in range(1, n):
        solver.add(s1[i] == Or(s1[i-1], literals[i]))
        solver.add(s2[i] == Or(s2[i-1], And(s1[i-1], literals[i])))
        # Overflow check for exactly 2 (cannot be >= 3)
        solver.add(Not(And(s2[i-1], literals[i])))

    # Final check: at least 2
    solver.add(s2[n-1])

# --- Totalizer Cardinality Constraints ---

def _totalizer_merge(solver: Solver, left_sum: List[Any], right_sum: List[Any], prefix: str = ""):
    merged_len = len(left_sum) + len(right_sum)
    merged_sum = [Bool(f"{prefix}_m_{i}") for i in range(merged_len)]
    
    for i in range(len(left_sum) + 1):
        for j in range(len(right_sum) + 1):
            target_idx = i + j - 1
            
            # Constraints based on: left >= i AND right >= j ==> merged >= i+j
            if i > 0 and j > 0:
                if i + j <= merged_len:
                    solver.add(Implies(And(left_sum[i-1], right_sum[j-1]), merged_sum[target_idx]))
            elif i > 0 and j == 0:
                if i <= merged_len:
                    solver.add(Implies(left_sum[i-1], merged_sum[i - 1]))
            elif j > 0 and i == 0:
                if j <= merged_len:
                    solver.add(Implies(right_sum[j-1], merged_sum[j - 1]))
    return merged_sum

def _build_totalizer(solver: Solver, literals: List[Any], prefix: str = ""):
    n = len(literals)
    if n == 0: return []
    nodes = [[l] for l in literals]
    idx = 0
    while len(nodes) > 1:
        next_nodes = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                merged = _totalizer_merge(solver, nodes[i], nodes[i+1], prefix=f"{prefix}_{idx}")
                next_nodes.append(merged)
                idx += 1
            else:
                next_nodes.append(nodes[i])
        nodes = next_nodes
    return nodes[0]

def totalizer_at_most_k(solver: Solver, literals: List[Any], k: int, prefix: str = ""):
    """Encodes Sum(literals) <= k using Totalizer."""
    n = len(literals)
    if k >= n: return
    if k < 0:
        for l in literals: solver.add(Not(l))
        return
    
    sum_vars = _build_totalizer(solver, literals, prefix)
    # sum_vars[k] represents sum >= k+1. We want NOT(sum >= k+1) -> sum <= k
    if k < len(sum_vars):
        solver.add(Not(sum_vars[k]))

def totalizer_at_least_k(solver: Solver, literals: List[Any], k: int, prefix: str = ""):
    """Encodes Sum(literals) >= k using Totalizer."""
    n = len(literals)
    if k <= 0: return
    if k > n:
        solver.add(BoolVal(False))
        return

    # sum_vars = _build_totalizer(solver, literals, prefix)
    # # sum_vars[k-1] represents sum >= k
    # if k > 0 and k-1 < len(sum_vars):
    #     solver.add(sum_vars[k-1])
    negated_literals = [Not(l) for l in literals]
    totalizer_at_most_k(solver, negated_literals, n - k, prefix=f"{prefix}_neg")

def totalizer_exactly_k(solver: Solver, literals: List[Any], k: int, prefix: str = ""):
    """Encodes Sum(literals) == k using Totalizer."""
    totalizer_at_most_k(solver, literals, k, prefix=f"{prefix}_le")
    totalizer_at_least_k(solver, literals, k, prefix=f"{prefix}_ge")



# --- Smart dispatcher ---
def smart_at_most_k(solver: Solver, literals: List[Any], k: int, prefix: str = ""):
    """
    Selects the most efficient encoding based on N (len(literals)) and k.
    """
    n = len(literals)
    if n <= k: return # Trivial

    # Case 1: Exactly/At Most One (Very common in STS)
    if k == 1:
        # Pairwise is best for propagation but O(N^2) clauses.
        # Threshold: ~12-15 variables is usually safe for Pairwise.
        if n <= 14 or n >= 20:
            at_most_1_pairwise(solver, literals)
        else:
            # For larger N, Sequential uses fewer clauses O(N) but adds vars.
            # Totalizer is overkill for k=1.
            sequential_at_most_k(solver, literals, 1, prefix)
            
    # Case 2: Small k (e.g. k=2 for "at most 2 games in period")
    elif k <= 3:
        # Pairwise is O(N^(k+1)), too expensive for N > 8 roughly.
        if n <= 8 or n >= 20:
            # Explicit exclusion still has strong propagation
            if k == 2: at_most_2_pairwise(solver, literals)
            else: sequential_at_most_k(solver, literals, k, prefix)
        else:
            # Sequential is robust here
            sequential_at_most_k(solver, literals, k, prefix)
            
    # Case 3: Large k or Optimization Context
    else:
        # Totalizer provides Arc Consistency and is efficient for 
        # bounds that might change or are large.
        totalizer_at_most_k(solver, literals, k, prefix)

def smart_exactly_k(solver: Solver, literals: List[Any], k: int, prefix: str = ""):
    """
    Seleziona l'encoding migliore per vincoli di cardinalità esatta.
    """
    n = len(literals)
    
    if k == 1:
        if n <= 14:
            exactly_1_pairwise(solver, literals)
        # Exactly-One è il vincolo più frequente. Pairwise è il più veloce a propagare.
        # Per N molto grandi (>20-24), potremmo considerare Heule, ma per ora Pairwise è sicuro.
        else:
            sequential_at_most_k(solver, literals, 1, prefix)
            at_least_1(solver, literals)
    else:
        # Per k > 1, Totalizer è superiore perché gestisce entrambi i bound (<= k e >= k)
        # con la stessa struttura ad albero.
        totalizer_exactly_k(solver, literals, k, prefix)

# --- Symmetry Breaking ---

def lex_lesseq(solver: Solver, list1: List[Any], list2: List[Any], prefix: str = ""):
    """Enforces list1 <=lex list2 using boolean logic."""
    n = len(list1)
    eq_prefix = [Bool(f"{prefix}_eq_{i}") for i in range(n)]

    # Base case
    solver.add(eq_prefix[0] == (list1[0] == list2[0]))
    
    # Recursive step
    for i in range(1, n):
        solver.add(eq_prefix[i] == And(eq_prefix[i-1], list1[i] == list2[i]))

    # Implication chain
    solver.add(Implies(list1[0], list2[0])) # i=0
    for i in range(1, n):
        solver.add(Implies(eq_prefix[i-1], Implies(list1[i], list2[i])))

# --- Balance Constraints ---

def add_balance_constraints_sequential(solver: Solver, n: int, matches: List, t1_plays_home: Dict, teams: List, max_imbalance: int):
    num_games = n - 1
    for t in teams:
        home_games = []
        for m in matches:
            if t == m[0]:
                home_games.append(t1_plays_home[m])
            elif t == m[1]:
                home_games.append(Not(t1_plays_home[m]))
        
        lower = math.ceil((num_games - max_imbalance) / 2)
        upper = math.floor((num_games + max_imbalance) / 2)
        
        sequential_at_least_k(solver, home_games, lower, prefix=f"bal_min_t{t}")
        sequential_at_most_k(solver, home_games, upper, prefix=f"bal_max_t{t}")

def add_balance_constraints_totalizer(solver: Solver, n: int, matches: List, t1_plays_home: Dict, teams: List, max_imbalance: int):
    num_games = n - 1
    for t in teams:
        home_games = []
        for m in matches:
            if t == m[0]:
                home_games.append(t1_plays_home[m])
            elif t == m[1]:
                home_games.append(Not(t1_plays_home[m]))
        
        lower = math.ceil((num_games - max_imbalance) / 2)
        upper = math.floor((num_games + max_imbalance) / 2)
        
        totalizer_at_least_k(solver, home_games, lower, prefix=f"bal_min_t{t}")
        totalizer_at_most_k(solver, home_games, upper, prefix=f"bal_max_t{t}")