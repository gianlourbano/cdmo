include "globals.mzn";

int: num_teams;
int: num_weeks;
int: num_periods;
array[Teams, Teams] of Weeks: week;

set of int: Teams = 1..num_teams;
set of int: Weeks = 0..num_weeks;
set of int: Periods = 0..num_periods;

%Decision variables
array[Teams, Teams] of var Periods: period;
array[Teams, Teams] of var bool: home;


%% -- BEGIN main constraints --

%Match validity
constraint forall(i,j in Teams where i!=j)(period[i,j]!=0);
constraint forall(i in Teams) (period[i,i] == 0);

%Period Simmetry
constraint forall(i,j in Teams where i<j)(period[i,j] == period[j,i]);

%Home-Away Consistency
constraint forall(i,j in Teams where i<j) (home[j,i] == 1- home[i,j]);

%Period Usage Constraint
constraint forall(t in Teams, p in 1..num_periods)(count_geq([period[t,j] | j in Teams where t != j], p, 2)::domain_propagation);

%Weekly Structure Constraint
constraint forall(w in 1..num_weeks, p in 1..num_periods )(sum([bool2int(week[i,j] == w /\ period[i,j] == p) | i,j in Teams where i < j]) = 1);

%% -- END main constraints --

solve satisfy;