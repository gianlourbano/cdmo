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

% Match validity
constraint forall(i,j in Teams where i!=j)(period[i,j]!=0);
constraint forall(i in Teams) (period[i,i] == 0);

%Period Simmetry
constraint forall(i,j in Teams where i<j)(period[i,j] == period[j,i]);

%Home-Away Consistency
constraint forall(i,j in Teams where i<j) (home[j,i] == 1- home[i,j]);

%Period Usage Constraint
constraint forall(t in Teams, p in 1..num_periods)(count_geq([period[t,j] | j in Teams where t != j], p, 2)::domain_propagation);

%Weekly Structure constraint
constraint forall(w in 1..num_weeks, p in 1..num_periods )(sum([bool2int(week[i,j] == w /\ period[i,j] == p) | i,j in Teams where i < j]) = 1);


%% -- END main constraints --

% -- BEGIN simmetry breaking --

%Period ordering
array[1..(num_teams div 2)] of var Periods: week1_periods = [ period[i,j] | i,j in Teams where i < j /\ week[i,j] == 1 ];
    
constraint increasing(week1_periods); %work better than strictly increasing

%Home-away simmetry
constraint forall(w in 1..(num_teams div 2)) (
  home[1, w + 1] == 1
);

constraint forall(w in ((num_teams div 2) + 1)..num_teams-1) (
  home[1, w + 1] == 0
);
% -- END simmetry breaking --

%% -- BEGIN  implicit constraints --

%%Deficient teams constraints

%1) exactly 2 teams play once in each period
constraint forall(p in 1..num_periods) (
  sum([bool2int(sum(j in Teams where j!=t)(bool2int(period[t,j] == p)) == 1) | t in Teams]) = 2
);

%2) exactly n_teams-2 teams play twice in each period
constraint forall(p in 1..num_periods) (
  sum([bool2int(sum(j in Teams where j!=t)(bool2int(period[t,j] == p)) == 2) | t in Teams]) = num_teams - 2
);

array[Teams, 1..num_periods] of var 0..(num_teams-1): occ; 
constraint forall(t in Teams, p in 1..num_periods) ( occ[t,p] = sum(j in Teams where j != t)( bool2int(period[t,j] == p) ) );

array[Teams, 1..num_periods] of var bool: deficient;
constraint forall(t in Teams, p in 1..num_periods) (
  deficient[t,p] <-> (occ[t,p] == 1)
);

%3) Every team is deficient in at most one period
constraint forall(t in Teams)(
    count_leq([deficient[t,p] | p in 1..num_periods], true, 1) 
);

constraint forall(t in Teams)(
  sum(p in 1..num_periods)( bool2int(deficient[t,p]) ) <= 1
);
% --- END implicit constraints --

% OPTIMIZATION
array[Teams] of var 0..num_weeks: home_count;
array[Teams] of var 0..num_weeks: away_count;

constraint forall(t in Teams) (
    home_count[t] = sum([bool2int(home[t,j] == 1) | j in Teams where j != t]));
constraint forall(t in Teams)(
    away_count[t] = sum([bool2int(home[t,j] == 0) | j in Teams where j != t])
  );
  
array[Teams] of var 0..num_weeks: imbalance;
constraint forall(t in Teams) (imbalance[t] = abs(home_count[t] - away_count[t]));

var int: total_imbalance = max(imbalance);
constraint total_imbalance >= 1;
constraint total_imbalance <= num_teams-1;

solve :: int_search([period[i,j] | i,j in Teams where i<j]++ [home[i,j]| i,j in Teams where i<j], dom_w_deg, indomain_min)::relax_and_reconstruct([period[i,j] | i,j in Teams where i<j] ++ [home[i,j]| i,j in Teams where i<j], 85) minimize total_imbalance;