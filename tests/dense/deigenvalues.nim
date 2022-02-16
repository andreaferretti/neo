# Copyright 2017 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest, neo/dense

proc run() =
  suite "matrix reductions for eigenvalue computations":
    test "balancing matrices":
      let
        a = matrix(@[
          @[3.0, 1.0, 0.0, 0.0],
          @[1.0, 0.0, 0.0, 0.0],
          @[2.0, -1.0, 1.5, 0.1],
          @[-1.0, 0.0, 1.1, 1.2],
        ])
        r = balance(a, BalanceOp.Permute)

      check(r.ilo == 1)
      check(r.ihi == 4)
    test "computing the upper Hessenberg form":
      let
        a = matrix(@[
          @[3.0, 1.0, 0.0, 0.0],
          @[1.0, 0.0, 0.0, 0.0],
          @[2.0, -1.0, 1.5, 0.1],
          @[-1.0, 0.0, 1.1, 1.2],
        ])
        r = hessenberg(a)

      check(r[0, 0] == 3)

  suite "computing eigenvalues":
    test "computing the eigenvalues alone":
      let
        a = matrix(@[
          @[3.0, 1.0, 0.0, 0.0],
          @[1.0, 0.0, 0.0, 0.0],
          @[2.0, -1.0, 1.5, 0.1],
          @[-1.0, 0.0, 1.1, 1.2],
        ])
        e = eigenvalues(a)

      check(e.img == @[0.0, 0.0, 0.0, 0.0])
    test "computing the eigenvalues of a known matrix":
      let
        a = matrix(@[
          @[2.0, 0.0, 1.0],
          @[0.0, 2.0, 0.0],
          @[1.0, 0.0, 2.0]
        ])
        e = eigenvalues(a)

      check(e.real == @[3.0, 1.0, 2.0])
      check(e.img == @[0.0, 0.0, 0.0])

    test "computing the Schur factorization":
      let
        a = matrix(@[
          @[2.0, 0.0, 1.0],
          @[0.0, 2.0, 0.0],
          @[1.0, 0.0, 2.0]
        ])
        s = schur(a)

      check(s.factorization == diag(3.0, 1.0, 2.0))
      check(s.eigenvalues.real == @[3.0, 1.0, 2.0])
      check(s.eigenvalues.img == @[0.0, 0.0, 0.0])

  suite "computing eigenvalues and eigenvectors of real symmetric matrix":
    test "computing the eigenvalues":
      let
        a = matrix(@[
          @[0.70794509, 0.3582868 , 0.18601989, 0.66848165],
          @[0.3582868 , 0.26329229, 0.85542206, 0.62635776],
          @[0.18601989, 0.85542206, 0.4399633 , 0.30754615],
          @[0.66848165, 0.62635776, 0.30754615, 0.89755355]])

      let (vals, vecs) = symeig(a)

      echo vecs.real

      let
        expected_vals = @[-0.564026237258954, 0.112267309803938, 0.643464411048214, 2.117048755152045]
        expected_vecs = @[@[ 0.029732015676491, -0.777482949372466,  0.597541851617029, 0.193855632482002],
                          @[-0.696167044814947, -0.015641070882148, -0.208509696605442, 0.686753601400669],
                          @[ 0.543396224349132, -0.39730786159535 , -0.655220561273828, 0.342860062651875],
                          @[0.46817517695895 , 0.487259769990589, 0.412496615830491, 0.610930816178531]]

      for i in 0..3:
        check abs(vals.real[i] - expected_vals[i]) < 1.0e-7

      for i in 0..3:
        for j in 0..3:
          check abs(vecs.real[i][j] - expected_vecs[i][j]) < 1.0e-7

      check(vals.img == @[0.0, 0.0, 0.0, 0.0])

run()
