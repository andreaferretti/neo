# Copyright 2016 UniCredit S.p.A.
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
  suite "vector accessors":
    test "reading vector length":
      let v = randomVector(10)
      check v.len == 10
    test "reading vector elements":
      let v = makeVector(5, proc(i: int): float64 = (3 * i - 2).float64)
      check v[0] == -2.0
      check v[1] == 1.0
      check v[2] == 4.0
      check v[3] == 7.0
      check v[4] == 10.0
    test "writing vector elements":
      var v = zeros(3)
      v[0] = -2.1
      v[1] = 1.0
      check v[0] == -2.1
      check v[1] == 1.0
    test "cloning vectors":
      var v = randomVector(5)
      let
        w = v.clone
        f = w[0]
      check v == w
      v[0] = v[0] + 1
      check w[0] == f
    test "mapping vectors":
      var v = vector([1.0, 2.0, 3.0, 4.0, 5.0])
      check v.map(proc(x: float64): float64 = 2 * x) ==
        vector([2.0, 4.0, 6.0, 8.0, 10.0])

  suite "matrix accessors":
    test "reading matrix dimensions":
      let m = randomMatrix(3, 7)
      check m.dim == (3, 7)
    test "reading matrix elements":
      let m = makeMatrix(2, 2, proc(i, j: int): float64 = (3 * i - 2 * j).float64)
      check m[0, 0] == 0.0
      check m[0, 1] == -2.0
      check m[1, 0] == 3.0
      check m[1, 1] == 1.0
    test "writing matrix elements":
      var m = zeros(3, 3)
      m[0, 2] = -2.1
      m[1, 1] = 1.0
      check m[0, 2] == -2.1
      check m[1, 1] == 1.0
    test "reading matrix rows":
      let
        m = makeMatrix(2, 2, proc(i, j: int): float64 = (3 * i - 2 * j).float64)
        r = m.row(1)
      check r[0] == 3.0
      check r[1] == 1.0
    test "reading matrix columns":
      let
        m = makeMatrix(2, 2, proc(i, j: int): float64 = (3 * i - 2 * j).float64)
        c = m.column(1)
      check c[0] == -2.0
      check c[1] == 1.0
    test "cloning matrices":
      var m = randomMatrix(5, 5)
      let
        n = m.clone
        f = n[2, 2]
      check m == n
      m[2, 2] = m[2, 2] + 1
      check n[2, 2] == f
    test "mapping matrices":
      let
        m = makeMatrix(2, 2, proc(i, j: int): float64 = (3 * i - 2 * j).float64)
        n = makeMatrix(2, 2, proc(i, j: int): float64 = (6 * i - 4 * j).float64)
      proc double(x: float64): float64 = 2 * x
      check m.map(double) == n

run()