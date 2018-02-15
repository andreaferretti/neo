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

suite "vectors and matrices on the stack":
  test "creating and accessing vectors on the stack":
    var
      data = [1'f64, 2, 3, 4, 5, 6]
      v = stackVector(data)

    check v.len == 6
    check v[4] == 5.0

    v[5] = -1.0
    check data[5] == -1.0

  test "operations on vectors on the stack":
    var
      data = [1'f64, 2, 3, 4, 5, 6]
      v = stackVector(data)
      w = vector(1.0, 2.0, 3.0, 1.0, 2.0, 3.0)

    check(v + w == vector(2.0, 4.0, 6.0, 5.0, 7.0, 9.0))
    v += w

    check v[1] == 4.0

    check v + w == w + v

  test "creating and accessing matrices on the stack":
    var
      data = [
        [1'f64, 2, 3],
        [4'f64, 5, 6],
        [7'f64, 8, 9]
      ]
      m = stackMatrix(data)

    check m.M == 3
    check m.N == 3
    check m[2, 1] == 6.0

    m[2, 0] = -1.0
    check data[0][2] == -1.0

  test "operations on matrices on the stack":
    var
      data = [
        [1'f64, 2, 3],
        [4'f64, 5, 6],
        [7'f64, 8, 9]
      ]
      m = stackMatrix(data)
      v = vector(1.0, 2.0, 3.0)

    check m * v == vector(30.0, 36.0, 42.0)

  test "slicing matrices on the stack":
    var
      data = [
        [1'f64, 2, 3],
        [4'f64, 5, 6],
        [7'f64, 8, 9]
      ]
      m = stackMatrix(data)

    check m[1 .. 2, 1 .. 2] == matrix(@[@[5.0, 8.0], @[6.0, 9.0]])
    check m.column(2) == vector(7.0, 8.0, 9.0)

  test "row major matrices on the stack":
    var
      data = [
        [1'f64, 2, 3],
        [4'f64, 5, 6],
        [7'f64, 8, 9]
      ]
      m = stackMatrix(data, rowMajor)

    check m.M == 3
    check m.N == 3
    check m[2, 1] == 8.0

    m[2, 0] = -1.0
    check data[2][0] == -1.0