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

import unittest, neo, neo/statics

proc run() =
  suite "row-major matrix/vector operations":
    test "multiplication of matrix and vector":
      let
        m = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ], order = rowMajor)
        v = vector([1.0, 3.0, 2.0, -2.0])
      check((m * v) == vector([7.0, 6.0, 5.0]))

  suite "row-major matrix operations":
    test "scalar matrix multiplication":
      let
        m1 = matrix([
          [1.0, 3.0],
          [2.0, 8.0],
          [-2.0, 3.0]
        ], order = rowMajor)
        m2 = matrix([
          [3.0, 9.0],
          [6.0, 24.0],
          [-6.0, 9.0]
        ], order = rowMajor)
      check(m1 * 3.0 == m2)
      check(3.0 * m1 == m2)
    test "mutating scalar multiplication":
      var m1 = matrix([
          [1.0, 3.0],
          [2.0, 8.0],
          [-2.0, 3.0]
        ], order = rowMajor)
      let m2 = matrix([
          [3.0, 9.0],
          [6.0, 24.0],
          [-6.0, 9.0]
        ], order = rowMajor)
      m1 *= 3.0
      check(m1 == m2)
    test "matrix sum":
      let
        m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ], order = rowMajor)
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ], order = rowMajor)
        m3 = matrix([
          [4.0, 1.0, 1.0, 0.0],
          [1.0, 2.0, 0.0, 1.0],
          [7.0, 3.0, 4.0, 6.0]
        ], order = rowMajor)
      check(m1 + m2 == m3)
    test "mutating matrix sum":
      var m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ], order = rowMajor)
      let
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ], order = rowMajor)
        m3 = matrix([
          [4.0, 1.0, 1.0, 0.0],
          [1.0, 2.0, 0.0, 1.0],
          [7.0, 3.0, 4.0, 6.0]
        ], order = rowMajor)
      m1 += m2
      check m1 == m3
    test "matrix difference":
      let
        m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ], order = rowMajor)
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ], order = rowMajor)
        m3 = matrix([
          [-2.0, -1.0, 3.0, -2.0],
          [-3.0, 0.0, 6.0, 1.0],
          [-1.0, 1.0, 0.0, 2.0]
        ], order = rowMajor)
      check(m1 - m2 == m3)
    test "mutating matrix sum":
      var m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ], order = rowMajor)
      let
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ], order = rowMajor)
        m3 = matrix([
          [-2.0, -1.0, 3.0, -2.0],
          [-3.0, 0.0, 6.0, 1.0],
          [-1.0, 1.0, 0.0, 2.0]
        ], order = rowMajor)
      m1 -= m2
      check m1 == m3
    test "matrix ℓ² norm":
      let m = matrix([
        [1.0, 1.0, 2.0],
        [3.0, 0.0, -7.0]
      ], order = rowMajor)
      check l_2(m) == 8.0
    test "matrix ℓ¹ norm":
      let m = matrix([
        [1.0, 1.0, 2.0],
        [3.0, 0.0, -7.0],
        [2.5, 3.1, -1.4]
      ], order = rowMajor)
      check l_1(m) == 21.0
    test "max and min of matrices":
      let m = matrix([
        [1.0, 1.0, 2.0],
        [3.0, 0.0, -7.0]
      ], order = rowMajor)
      check max(m) == 3.0
      check min(m) == -7.0
    test "matrix multiplication":
      let
        m1 = matrix([
          [1.0, 1.0, 2.0, -3.0],
          [3.0, 0.0, -7.0, 2.0]
        ], order = rowMajor)
        m2 = matrix([
          [1.0, 1.0, 2.0],
          [3.0, 1.0, -5.0],
          [-1.0, -1.0, 2.0],
          [4.0, 2.0, 3.0]
        ], order = rowMajor)
        m3 = matrix([
          [-10.0, -6.0, -8.0],
          [18.0, 14.0, -2.0]
        ], order = rowMajor)
      check(m1 * m2 == m3)
    test "matrix Hadamard multiplication":
      let
        m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ], order = rowMajor)
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ], order = rowMajor)
        m3 = matrix([
          [3.0, 0.0, -2.0, -1.0],
          [-2.0, 1.0, -9.0, 0.0],
          [12.0, 2.0, 4.0, 8.0]
        ], order = rowMajor)
      check((m1 |*| m2) == m3)

run()