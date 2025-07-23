% -----------------------------------------------------------------------------
% MiniZinc model for the Sports Tournament Scheduling (STS) problem. (v2)
%
% This version corrects the overly restrictive symmetry-breaking constraints.
% -----------------------------------------------------------------------------

include "globals.mzn";

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. PARAMETERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int: n; % The number of teams (must be an even integer)

set of int: TEAMS = 1..n;
set of int: WEEKS = 1..n-1;
set of int: PERIODS = 1..n div 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. DECISION VARIABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

array[WEEKS, PERIODS] of var TEAMS: home;
array[WEEKS, PERIODS] of var TEAMS: away;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. CONSTRAINTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Constraint 1: Every team plays exactly once per week.
constraint forall(w in WEEKS) (
    alldifferent([home[w, p] | p in PERIODS] ++ [away[w, p] | p in PERIODS])
);

% Constraint 2: Every team plays every other team exactly once.
array[WEEKS, PERIODS] of var TEAMS: min_team;
array[WEEKS, PERIODS] of var TEAMS: max_team;

constraint forall(w in WEEKS, p in PERIODS)(
    min_team[w,p] = min(home[w,p], away[w,p]) /\
    max_team[w,p] = max(home[w,p], away[w,p])
);

constraint alldifferent([ min_team[w,p] * n + max_team[w,p] | w in WEEKS, p in PERIODS ]);


% Constraint 3: Every team plays at most twice in the same period.
constraint forall(t in TEAMS, p in PERIODS) (
    sum(w in WEEKS) (bool2int(home[w, p] = t) + bool2int(away[w, p] = t)) <= 2
);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4. SYMMETRY BREAKING (Corrected)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Optional: To speed up search for larger 'n', you can break symmetry
% on the weeks by enforcing lexicographical ordering. This means the schedule
% for Week 1 must come "before" the schedule for Week 2, etc.
% This is effective but can make the model slower for small 'n'.
%
% constraint forall(w in 1..n-2) (
%     [home[w,p] | p in PERIODS] <= lex([home[w+1,p] | p in PERIODS])
% );

% constraint home[1, 1] = 1;
% constraint away[1, 1] = 2;

constraint forall(w in 1..n-2) (
    lex_less(
      [home[w,p] | p in PERIODS] ++ [away[w,p] | p in PERIODS],
      [home[w+1,p] | p in PERIODS] ++ [away[w+1,p] | p in PERIODS]
    )
);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5. OPTIONAL OPTIMIZATION OBJECTIVE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (Content is unchanged, can be used as before)
% ...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6. SEARCH STRATEGY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tell the solver how to make decisions.
% - first_fail: choose the variable with the smallest remaining domain.
% - indomain_min: try the smallest value in the domain first.
ann: search_strategy = int_search(
    [home[w,p] | w in WEEKS, p in PERIODS] ++ [away[w,p] | w in WEEKS, p in PERIODS],
    first_fail,
    indomain_min,
    complete
);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7. SOLVE STATEMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

solve :: search_strategy satisfy;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 6. SOLVE STATEMENT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% solve satisfy;
% solve minimize max_imbalance;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7. OUTPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (Content is unchanged, will work now)
output ["--- SCHEDULE for n = ", show(n), " ---\n\n"];
output [
    if p = 1 then "Week " ++ show(w) ++ ":\n" else "" endif ++
    "  Period " ++ show(p) ++ ": " ++ show(home[w,p]) ++ " vs " ++ show(away[w,p]) ++ "\n"
    | w in WEEKS, p in PERIODS
];