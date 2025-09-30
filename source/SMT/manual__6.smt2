; =====================================================================
; SMT-LIB 2 Formulation for the STS Period Assignment Subproblem
;
; - Instance: n=6 teams
; - Presolver has already determined all matchups for all 5 weeks.
; - The SMT solver's only task is to assign each game to a period (1, 2, or 3).
; =====================================================================

(set-logic QF_LIA) ; Quantifier-Free Linear Integer Arithmetic
(set-option :produce-models true)
(set-option :pp.decimal true) ; Optional: For cleaner output formatting

; ---------------------------------------------------------------------
; Decision Variables
; ---------------------------------------------------------------------
; We declare one integer variable for each of the 15 pre-computed games.
; The variable represents the period (1, 2, or 3) that the game is played in.
; Naming convention: p_W_T1_T2 where W is the week, T1/T2 are the teams.

; --- Week 1 Games: (6,1), (2,5), (3,4) ---
(declare-fun p_1_6_1 () Int)
(declare-fun p_1_2_5 () Int)
(declare-fun p_1_3_4 () Int)

; --- Week 2 Games: (6,2), (3,1), (4,5) ---
(declare-fun p_2_6_2 () Int)
(declare-fun p_2_3_1 () Int)
(declare-fun p_2_4_5 () Int)

; --- Week 3 Games: (6,3), (4,2), (5,1) ---
(declare-fun p_3_6_3 () Int)
(declare-fun p_3_4_2 () Int)
(declare-fun p_3_5_1 () Int)

; --- Week 4 Games: (6,4), (5,3), (1,2) ---
(declare-fun p_4_6_4 () Int)
(declare-fun p_4_5_3 () Int)
(declare-fun p_4_1_2 () Int)

; --- Week 5 Games: (6,5), (1,4), (2,3) ---
(declare-fun p_5_6_5 () Int)
(declare-fun p_5_1_4 () Int)
(declare-fun p_5_2_3 () Int)


; ---------------------------------------------------------------------
; Constraints
; ---------------------------------------------------------------------

; --- Constraint 1: Domain Constraint ---
; All period variables must be between 1 and 3 (inclusive).
(assert (and
    (>= p_1_6_1 1) (<= p_1_6_1 3)
    (>= p_1_2_5 1) (<= p_1_2_5 3)
    (>= p_1_3_4 1) (<= p_1_3_4 3)
    (>= p_2_6_2 1) (<= p_2_6_2 3)
    (>= p_2_3_1 1) (<= p_2_3_1 3)
    (>= p_2_4_5 1) (<= p_2_4_5 3)
    (>= p_3_6_3 1) (<= p_3_6_3 3)
    (>= p_3_4_2 1) (<= p_3_4_2 3)
    (>= p_3_5_1 1) (<= p_3_5_1 3)
    (>= p_4_6_4 1) (<= p_4_6_4 3)
    (>= p_4_5_3 1) (<= p_4_5_3 3)
    (>= p_4_1_2 1) (<= p_4_1_2 3)
    (>= p_5_6_5 1) (<= p_5_6_5 3)
    (>= p_5_1_4 1) (<= p_5_1_4 3)
    (>= p_5_2_3 1) (<= p_5_2_3 3)
))


; --- Constraint 2: One match per slot per week ---
; Within each week, all games must be assigned to different periods.
(assert (distinct p_1_6_1 p_1_2_5 p_1_3_4))
(assert (distinct p_2_6_2 p_2_3_1 p_2_4_5))
(assert (distinct p_3_6_3 p_3_4_2 p_3_5_1))
(assert (distinct p_4_6_4 p_4_5_3 p_4_1_2))
(assert (distinct p_5_6_5 p_5_1_4 p_5_2_3))


; --- Constraint 3: Each team plays at most twice in the same period ---
; For each team T and for each period P, the sum of its appearances in period P must be <= 2.
; We use (ite (= variable P) 1 0) which means: if (variable == P) then 1 else 0.

; -- For Team 1: Plays in games (6,1), (3,1), (5,1), (1,2), (1,4)
(assert (<= (+ (ite (= p_1_6_1 1) 1 0) (ite (= p_2_3_1 1) 1 0) (ite (= p_3_5_1 1) 1 0) (ite (= p_4_1_2 1) 1 0) (ite (= p_5_1_4 1) 1 0)) 2))
(assert (<= (+ (ite (= p_1_6_1 2) 1 0) (ite (= p_2_3_1 2) 1 0) (ite (= p_3_5_1 2) 1 0) (ite (= p_4_1_2 2) 1 0) (ite (= p_5_1_4 2) 1 0)) 2))
(assert (<= (+ (ite (= p_1_6_1 3) 1 0) (ite (= p_2_3_1 3) 1 0) (ite (= p_3_5_1 3) 1 0) (ite (= p_4_1_2 3) 1 0) (ite (= p_5_1_4 3) 1 0)) 2))

; -- For Team 2: Plays in games (2,5), (6,2), (4,2), (1,2), (2,3)
(assert (<= (+ (ite (= p_1_2_5 1) 1 0) (ite (= p_2_6_2 1) 1 0) (ite (= p_3_4_2 1) 1 0) (ite (= p_4_1_2 1) 1 0) (ite (= p_5_2_3 1) 1 0)) 2))
(assert (<= (+ (ite (= p_1_2_5 2) 1 0) (ite (= p_2_6_2 2) 1 0) (ite (= p_3_4_2 2) 1 0) (ite (= p_4_1_2 2) 1 0) (ite (= p_5_2_3 2) 1 0)) 2))
(assert (<= (+ (ite (= p_1_2_5 3) 1 0) (ite (= p_2_6_2 3) 1 0) (ite (= p_3_4_2 3) 1 0) (ite (= p_4_1_2 3) 1 0) (ite (= p_5_2_3 3) 1 0)) 2))

; -- For Team 3: Plays in games (3,4), (3,1), (6,3), (5,3), (2,3)
(assert (<= (+ (ite (= p_1_3_4 1) 1 0) (ite (= p_2_3_1 1) 1 0) (ite (= p_3_6_3 1) 1 0) (ite (= p_4_5_3 1) 1 0) (ite (= p_5_2_3 1) 1 0)) 2))
(assert (<= (+ (ite (= p_1_3_4 2) 1 0) (ite (= p_2_3_1 2) 1 0) (ite (= p_3_6_3 2) 1 0) (ite (= p_4_5_3 2) 1 0) (ite (= p_5_2_3 2) 1 0)) 2))
(assert (<= (+ (ite (= p_1_3_4 3) 1 0) (ite (= p_2_3_1 3) 1 0) (ite (= p_3_6_3 3) 1 0) (ite (= p_4_5_3 3) 1 0) (ite (= p_5_2_3 3) 1 0)) 2))

; -- For Team 4: Plays in games (3,4), (4,5), (4,2), (6,4), (1,4)
(assert (<= (+ (ite (= p_1_3_4 1) 1 0) (ite (= p_2_4_5 1) 1 0) (ite (= p_3_4_2 1) 1 0) (ite (= p_4_6_4 1) 1 0) (ite (= p_5_1_4 1) 1 0)) 2))
(assert (<= (+ (ite (= p_1_3_4 2) 1 0) (ite (= p_2_4_5 2) 1 0) (ite (= p_3_4_2 2) 1 0) (ite (= p_4_6_4 2) 1 0) (ite (= p_5_1_4 2) 1 0)) 2))
(assert (<= (+ (ite (= p_1_3_4 3) 1 0) (ite (= p_2_4_5 3) 1 0) (ite (= p_3_4_2 3) 1 0) (ite (= p_4_6_4 3) 1 0) (ite (= p_5_1_4 3) 1 0)) 2))

; -- For Team 5: Plays in games (2,5), (4,5), (5,1), (5,3), (6,5)
(assert (<= (+ (ite (= p_1_2_5 1) 1 0) (ite (= p_2_4_5 1) 1 0) (ite (= p_3_5_1 1) 1 0) (ite (= p_4_5_3 1) 1 0) (ite (= p_5_6_5 1) 1 0)) 2))
(assert (<= (+ (ite (= p_1_2_5 2) 1 0) (ite (= p_2_4_5 2) 1 0) (ite (= p_3_5_1 2) 1 0) (ite (= p_4_5_3 2) 1 0) (ite (= p_5_6_5 2) 1 0)) 2))
(assert (<= (+ (ite (= p_1_2_5 3) 1 0) (ite (= p_2_4_5 3) 1 0) (ite (= p_3_5_1 3) 1 0) (ite (= p_4_5_3 3) 1 0) (ite (= p_5_6_5 3) 1 0)) 2))

; -- For Team 6: Plays in games (6,1), (6,2), (6,3), (6,4), (6,5)
(assert (<= (+ (ite (= p_1_6_1 1) 1 0) (ite (= p_2_6_2 1) 1 0) (ite (= p_3_6_3 1) 1 0) (ite (= p_4_6_4 1) 1 0) (ite (= p_5_6_5 1) 1 0)) 2))
(assert (<= (+ (ite (= p_1_6_1 2) 1 0) (ite (= p_2_6_2 2) 1 0) (ite (= p_3_6_3 2) 1 0) (ite (= p_4_6_4 2) 1 0) (ite (= p_5_6_5 2) 1 0)) 2))
(assert (<= (+ (ite (= p_1_6_1 3) 1 0) (ite (= p_2_6_2 3) 1 0) (ite (= p_3_6_3 3) 1 0) (ite (= p_4_6_4 3) 1 0) (ite (= p_5_6_5 3) 1 0)) 2))


; ---------------------------------------------------------------------
; Solve and get the model
; ---------------------------------------------------------------------
(check-sat)
(get-model)