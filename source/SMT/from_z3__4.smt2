; benchmark generated from python API
(set-info :status unknown)
(declare-fun p_1_4_1 () Int)
(declare-fun p_1_2_3 () Int)
(declare-fun p_2_4_2 () Int)
(declare-fun p_2_3_1 () Int)
(declare-fun p_3_3_4 () Int)
(declare-fun p_3_1_2 () Int)
(assert
 (>= p_1_4_1 1))
(assert
 (<= p_1_4_1 2))
(assert
 (>= p_1_2_3 1))
(assert
 (<= p_1_2_3 2))
(assert
 (>= p_2_4_2 1))
(assert
 (<= p_2_4_2 2))
(assert
 (>= p_2_3_1 1))
(assert
 (<= p_2_3_1 2))
(assert
 (>= p_3_3_4 1))
(assert
 (<= p_3_3_4 2))
(assert
 (>= p_3_1_2 1))
(assert
 (<= p_3_1_2 2))
(assert
 (= p_1_2_3 1))
(assert
 (= p_1_4_1 2))
(assert
 (let (($x25 (= p_1_2_3 1)))
 (let (($x27 (= p_1_4_1 1)))
 ((_ pbeq 1 1 1) $x27 $x25))))
(assert
 (let (($x29 (= p_1_2_3 2)))
 (let (($x26 (= p_1_4_1 2)))
 ((_ pbeq 1 1 1) $x26 $x29))))
(assert
 (let (($x32 (= p_2_3_1 1)))
 (let (($x31 (= p_2_4_2 1)))
 ((_ pbeq 1 1 1) $x31 $x32))))
(assert
 (let (($x35 (= p_2_3_1 2)))
 (let (($x34 (= p_2_4_2 2)))
 ((_ pbeq 1 1 1) $x34 $x35))))
(assert
 (let (($x38 (= p_3_1_2 1)))
 (let (($x37 (= p_3_3_4 1)))
 ((_ pbeq 1 1 1) $x37 $x38))))
(assert
 (let (($x41 (= p_3_1_2 2)))
 (let (($x40 (= p_3_3_4 2)))
 ((_ pbeq 1 1 1) $x40 $x41))))
(assert
 (let (($x38 (= p_3_1_2 1)))
 (let (($x32 (= p_2_3_1 1)))
 (let (($x27 (= p_1_4_1 1)))
 ((_ at-most 2) $x27 $x32 $x38)))))
(assert
 (let (($x41 (= p_3_1_2 2)))
 (let (($x35 (= p_2_3_1 2)))
 (let (($x26 (= p_1_4_1 2)))
 ((_ at-most 2) $x26 $x35 $x41)))))
(assert
 (let (($x38 (= p_3_1_2 1)))
 (let (($x31 (= p_2_4_2 1)))
 (let (($x25 (= p_1_2_3 1)))
 ((_ at-most 2) $x25 $x31 $x38)))))
(assert
 (let (($x41 (= p_3_1_2 2)))
 (let (($x34 (= p_2_4_2 2)))
 (let (($x29 (= p_1_2_3 2)))
 ((_ at-most 2) $x29 $x34 $x41)))))
(assert
 (let (($x37 (= p_3_3_4 1)))
 (let (($x32 (= p_2_3_1 1)))
 (let (($x25 (= p_1_2_3 1)))
 ((_ at-most 2) $x25 $x32 $x37)))))
(assert
 (let (($x40 (= p_3_3_4 2)))
 (let (($x35 (= p_2_3_1 2)))
 (let (($x29 (= p_1_2_3 2)))
 ((_ at-most 2) $x29 $x35 $x40)))))
(assert
 (let (($x37 (= p_3_3_4 1)))
 (let (($x31 (= p_2_4_2 1)))
 (let (($x27 (= p_1_4_1 1)))
 ((_ at-most 2) $x27 $x31 $x37)))))
(assert
 (let (($x40 (= p_3_3_4 2)))
(let (($x34 (= p_2_4_2 2)))
(let (($x26 (= p_1_4_1 2)))
((_ at-most 2) $x26 $x34 $x40)))))
(check-sat)
