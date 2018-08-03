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
  suite "linear system solving":
    test "matrix-matrix solver":
      let
        a = matrix([
          [3.0, 1.0],
          [1.0, -2.0]
        ])
        b = matrix([
          [1.0],
          [0.0]
        ])
        x = solve(a, b)
        expected = matrix([
          [2.0 / 7.0],
          [1.0 / 7.0]
        ])
      check expected =~ x

    test "matrix-matrix solve operator":
      let
        a = matrix([
          [3.0, 1.0],
          [1.0, -2.0]
        ])
        b = matrix([
          [1.0],
          [0.0]
        ])
        x = a \ b
        expected = matrix([
          [2.0 / 7.0],
          [1.0 / 7.0]
        ])
      check expected =~ x

    test "singular matrix error":
      let
        a = matrix([
          [2.0, 2.0],
          [1.0, 1.0]
        ])
        b = matrix([
          [1.0],
          [0.0]
        ])
      expect FloatingPointError:
        discard solve(a, b)

    test "matrix inverse":
      let
        a = matrix([
          [4.0, 3.0],
          [3.0, 2.0]
        ])
        expected = matrix([
          [-2.0, 3.0],
          [3.0, -4.0]
        ])
        ainv = inv(a)
      check expected =~ ainv

    test "matrix-vector solver":
      let
        a = matrix([
          [3.0, 1.0],
          [1.0, -2.0]
        ])
        b = vector([1.0, 0.0])
        x = solve(a, b)
        expected = vector([2.0/7.0, 1.0/7.0])
      check expected =~ x

    test "matrix-vector solve operator":
      let
        a = matrix([
          [3.0, 1.0],
          [1.0, -2.0]
        ])
        b = vector([1.0, 0.0])
        x = a \ b
        expected = vector([2.0/7.0, 1.0/7.0])
      check expected =~ x

    test "matrix-vector singular matrix error":
      let
        a = matrix([
          [0.0, 0.0],
          [0.0, 0.0]
        ])
        b = vector([1.0, 0.0])
      expect FloatingPointError:
        discard solve(a, b)

run()