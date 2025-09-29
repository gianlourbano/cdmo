include "globals.mzn";

int: num_teams;
int: num_weeks;
int: num_periods;
array[Teams, Teams] of Weeks: week;

set of int: Teams = 1..num_teams;
set of int: Weeks = 0..num_weeks;
set of int: Periods = 0..num_periods;

% Decision variables
array[Teams, Teams] of var Periods: period;
array[Teams, Teams] of var bool: home;

%% -- BEGIN main constraints --

% enforce symmetry of the tables
constraint forall(i,j in Teams where i!=j)(period[i,j]!=0);
constraint forall(i in Teams) (period[i,i] == 0);
constraint forall(i,j in Teams where i<j)(period[i,j] == period[j,i]);
constraint forall(i,j in Teams where i<j) (home[j,i] == 1- home[i,j]);

% Each team plays at most twice in the same period
constraint forall(t in Teams, p in 1..num_periods)(count_geq([period[t,j] | j in Teams where t != j], p, 2)::domain_propagation);

/*
constraint forall(t in Teams)(
  let {
    array[1..num_periods] of var 1..2: occ
  } in
    global_cardinality_closed(
      [period[t,j] | j in Teams where j != t],
      [p | p in 1..num_periods],
      occ
    )
);

*/  
% Consistent fill between week and slot - each week exactly n/2 matches
constraint forall(w in 1..num_weeks, p in 1..num_periods )(   sum([bool2int(week[i,j] == w /\ period[i,j] == p) | i,j in Teams where i < j]) = 1);
/*
constraint forall(w in 1..num_weeks)(
   alldifferent([ period[i,j] | i,j in Teams where i < j /\ week[i,j] == w ])::domain_propagation);
*/

%% -- END main constraints --

% -- BEGIN simmetry breaking --
/*
% breaking team exchange simmetry
array[Teams, 1..(num_teams - 1)] of var Periods: period_no_diagonal;

constraint forall(i in Teams)(
    period_no_diagonal[i, ..] = [period[i, j] | j in Teams where j != i]
);

%constraint lex2(period_no_diagonal);

constraint forall(i in 1..num_teams-1)(
  lex_lesseq(period_no_diagonal[i, ..], period_no_diagonal[i+1, ..]) %better

);

*/
% Create an array of the periods of the first week
array[1..(num_teams div 2)] of var Periods: week1_periods =
  [ period[i,j] | i,j in Teams where i < j /\ week[i,j] == 1 ];

constraint increasing(week1_periods); %work better than strictly increasing

% fix the games team 1 plays at home and those it plays away
constraint forall(i, j in Teams where i < j) (
  home[i, j] == 1);
% -- END simmetry breaking --

%% -- BEGIN  implicit constraints --

%%  deficient teams constraint

% exactly 2 teams play once in each period
constraint forall(p in 1..num_periods) (
  sum([bool2int(sum(j in Teams where j!=t)(bool2int(period[t,j] == p)) == 1) | t in Teams]) = 2
);

% exactly n_teams-2 teams play twice in each period
constraint forall(p in 1..num_periods) (
  sum([bool2int(sum(j in Teams where j!=t)(bool2int(period[t,j] == p)) == 2) | t in Teams]) = num_teams - 2
);

/*
% occ[t,p] = numero di partite che il team t gioca nel periodo p
% occ[t,p] = numero di partite che il team t gioca nel periodo p
array[Teams, 1..num_periods] of var 1..2: occ;  % dominio ristretto 0..2

constraint forall(t in Teams, p in 1..num_periods) (
  occ[t,p] = sum(j in Teams where j != t)( bool2int(period[t,j] == p) )
);


constraint forall(p in 1..num_periods) (
  distribute(
    [2, num_teams - 2],
    [1,2],
    [occ[t,p] | t in Teams]   % lista delle occorrenze
                           % valori possibili
           % conteggi esatti
  )
);

*/

%% extension deficient
array[Teams, 1..num_periods] of var 0..(num_teams-1): occ; 

constraint forall(t in Teams, p in 1..num_periods) ( occ[t,p] = sum(j in Teams where j != t)( bool2int(period[t,j] == p) ) );

% Deficient[t,p] = 1 iff occ[t,p] == 1
array[Teams, 1..num_periods] of var bool: deficient;
constraint forall(t in Teams, p in 1..num_periods) (
  deficient[t,p] <-> (occ[t,p] == 1)
);

/*
constraint forall(t in Teams)(
  sum(p in 1..num_periods)( bool2int(deficient[t,p]) ) <= 1
);
*/

% Every team is deficient in at most one period
constraint forall(t in Teams)(
    count_leq([deficient[t,p] | p in 1..num_periods], true, 1)
);
/*
constraint forall(t in Teams) (
    sum(p in 1..num_periods)(bool2int(deficient[t,p])) = 1 %better for n=18 (2 sec) but worse for n=20 (96 sec)
); 
*/
% --- END implicit constraints --


solve satisfy; %best one
%results for n=18
%solve :: int_search([period[i,j] | i,j in Teams where i<j] ++ [home[i,j] | i,j in Teams where i<j], input_order, indomain_random) satisfy; Not supported by chuffed

%solve :: int_search([period[i,j] | i,j in Teams where i<j] ++ [home[i,j] | i,j in Teams where i<j], first_fail, indomain_min) satisfy; 60s

%solve :: int_search([period[i,j] | i,j in Teams where i<j]++[home[i,j]| i,j in Teams where i<j], dom_w_deg, indomain_min) :: restart_luby(50) satisfy;
%solve :: int_search([period[i,j] | i,j in Teams where i<j]++ [home[i,j]| i,j in Teams where i<j], dom_w_deg, indomain_random)::relax_and_reconstruct([period[i,j] | i,j in Teams where i<j] ++ [home[i,j]| i,j in Teams where i<j], 85) satisfy;
%solve::seq_search([
%int_search([period[i,j] | i,j in Teams where i<j], first_fail, indomain_split), 
%int_search([home[i,j] | i,j in Teams where i<j], first_fail, indomain_split)]) satisfy;
/*
solve::seq_search([
  int_search([period[i,j] | i,j in Teams where i<j], random_order, indomain_split), 
  int_search([home[i,j] | i,j in Teams where i<j], random_order, indomain_split)]) :: restart_luby(50) satisfy;
 */