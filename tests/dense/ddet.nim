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
  suite "trace and determinant computations":
    test "trace of a matrix":
      let a = makeMatrixIJ(int, 3, 3, i + i * j - 1)

      check(tr(a) == 5)
    
    test "determinant of a 3x3 matrix":
      let a = matrix(@[
        @[-1.0, -1.0, 0.0],
        @[ 0.0,  1.0, 2.0],
        @[ 1.0,  3.0, 5.0]
      ])

      check((det(a) + -1) < 1e-6)
    
    test "determinant of a 2x2 matrix":
      let a = matrix(@[
        @[1.0, 1.0],
        @[-1.0,  1.0]
      ])

      check(det(a) == 2.0)

run()