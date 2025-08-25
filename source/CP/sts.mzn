% Sports Tournament Scheduling (STS) Problem - MiniZinc Model
% Authors: CDMO Project Team
% Date: 2025

include "globals.mzn";

% Parameters
int: n;  % number of teams (must be even)
int: weeks = n - 1;
int: periods = n div 2;

% Ensure n is even and >= 4
constraint n >= 4;
constraint n mod 2 = 0;

% Decision variables
% schedule[w][p][s] = team playing in week w, period p, slot s (1=home, 2=away)
array[1..weeks, 1..periods, 1..2] of var 1..n: schedule;

% Constraints

% 1. Every team plays with every other team only once
constraint forall(t1, t2 in 1..n where t1 < t2) (
    sum(w in 1..weeks, p in 1..periods) (
        bool2int((schedule[w][p][1] = t1 /\ schedule[w][p][2] = t2) \/
                 (schedule[w][p][1] = t2 /\ schedule[w][p][2] = t1))
    ) = 1
);

% 2. Every team plays once a week
constraint forall(w in 1..weeks, t in 1..n) (
    sum(p in 1..periods, s in 1..2) (
        bool2int(schedule[w][p][s] = t)
    ) = 1
);

% 3. Every team plays at most twice in the same period over the tournament
constraint forall(p in 1..periods, t in 1..n) (
    sum(w in 1..weeks, s in 1..2) (
        bool2int(schedule[w][p][s] = t)
    ) <= 2
);

% 4. No team plays against itself
constraint forall(w in 1..weeks, p in 1..periods) (
    schedule[w][p][1] != schedule[w][p][2]
);

% Optional: Symmetry breaking constraints
% Fix team 1 to always play at home in the first period of the first week
constraint schedule[1][1][1] = 1;

% Team 1 plays against teams 2, 3, ..., n in some order
constraint forall(w in 1..weeks) (
    exists(p in 1..periods) (
        (schedule[w][p][1] = 1 /\ schedule[w][p][2] = w + 1) \/
        (schedule[w][p][1] = w + 1 /\ schedule[w][p][2] = 1)
    )
);

% Solve
solve satisfy;

% Output in the required format for JSON conversion
output [
    "{\n",
    "  \"schedule\": [\n"
] ++
[
    if p = 1 /\ w = 1 then "    ["
    elseif p = 1 then ",\n    ["
    else ""
    endif ++
    if p > 1 then ", " else "" endif ++
    "[" ++ show(schedule[w][p][1]) ++ ", " ++ show(schedule[w][p][2]) ++ "]"
    | w in 1..weeks, p in 1..periods
] ++
[
    if p = periods /\ w = weeks then "]\n"
    elseif p = periods then "]"
    else ""
    endif
    | w in 1..weeks, p in 1..periods
] ++
[
    "  ]\n",
    "}\n"
];