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