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
  suite "shared vectors and matrices":
    test "creating and accessing shared vectors":
      var v = sharedVector(5, float64)
      defer:
        dealloc(v)

      check v.len == 5

      v[3] = -1.0
      v[2] = 3.0
      check v[3] == -1.0
      check v.sum == 2.0

    test "operations on shared vectors":
      var v = sharedVector(6, float64)
      defer:
        dealloc(v)
      let w = vector(1.0, 2.0, 3.0, 1.0, 2.0, 3.0)
      for i in 0 .. 5:
        v[i] = (i + 1).float64

      check(v + w == vector(2.0, 4.0, 6.0, 5.0, 7.0, 9.0))
      v += w

      check v[1] == 4.0

      check v + w == w + v

    test "creating and accessing shared matrices":
      let data = [
          [1'f64, 2, 3],
          [4'f64, 5, 6],
          [7'f64, 8, 9]
        ]
      var m = sharedMatrix(3, 3, float64)
      defer:
        dealloc(m)
      for i in 0 .. 2:
        for j in 0 .. 2:
          m[i, j] = data[j][i]

      check m.M == 3
      check m.N == 3
      check m[2, 1] == 6.0

      m[2, 0] = -1.0
      check m[2, 0] == -1.0

    test "operations on shared matrices":
      let
        data = [
          [1'f64, 2, 3],
          [4'f64, 5, 6],
          [7'f64, 8, 9]
        ]
        v = vector(1.0, 2.0, 3.0)
      var m = sharedMatrix(3, 3, float64)
      defer:
        dealloc(m)
      for i in 0 .. 2:
        for j in 0 .. 2:
          m[i, j] = data[j][i]

      check m * v == vector(30.0, 36.0, 42.0)

    test "slicing shared matrices":
      let
        data = [
          [1'f64, 2, 3],
          [4'f64, 5, 6],
          [7'f64, 8, 9]
        ]
        v = vector(1.0, 2.0, 3.0)
      var m = sharedMatrix(3, 3, float64)
      defer:
        dealloc(m)
      for i in 0 .. 2:
        for j in 0 .. 2:
          m[i, j] = data[j][i]

      check m * v == vector(30.0, 36.0, 42.0)

      check m[1 .. 2, 1 .. 2] == matrix(@[@[5.0, 8.0], @[6.0, 9.0]])
      check m.column(2) == vector(7.0, 8.0, 9.0)

    test "row major shared matrices":
      let
        data = [
          [1'f64, 2, 3],
          [4'f64, 5, 6],
          [7'f64, 8, 9]
        ]
      var m = sharedMatrix(3, 3, float64, rowMajor)
      defer:
        dealloc(m)
      for i in 0 .. 2:
        for j in 0 .. 2:
          m[i, j] = data[i][j]

      check m.M == 3
      check m.N == 3
      check m[2, 1] == 8.0

      m[2, 0] = -1.0
      check m[2, 0] == -1.0

run()