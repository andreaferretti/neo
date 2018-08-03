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
  suite "vector and matrix equality":
    test "strict vector equality":
      let
        u = vector([1.0, 2.0, 3.0, 4.0])
        v = vector([1.0, 2.0, 3.0, 4.0])
        w = vector([1.0, 3.0, 3.0, 4.0])
      check u == v
      check v != w
    test "strict 32-bit vector equality":
      let
        u = vector([1'f32, 2'f32, 3'f32, 4'f32])
        v = vector([1'f32, 2'f32, 3'f32, 4'f32])
        w = vector([1'f32, 3'f32, 3'f32, 4'f32])
      check u == v
      check v != w
    test "strict matrix equality":
      let
        m = makeMatrix(3, 5, proc(i, j: int): float64 = (i + 3 * j).float64)
        n = makeMatrix(3, 5, proc(i, j: int): float64 = (i + 3 * j).float64)
        p = makeMatrix(3, 5, proc(i, j: int): float64 = (i - 2 * j).float64)
      check m == n
      check n != p
    test "strict 32-bit matrix equality":
      let
        m = makeMatrix(3, 5, proc(i, j: int): float32 = (i + 3 * j).float32)
        n = makeMatrix(3, 5, proc(i, j: int): float32 = (i + 3 * j).float32)
        p = makeMatrix(3, 5, proc(i, j: int): float32 = (i - 2 * j).float32)
      check m == n
      check n != p
    test "approximate vector equality":
      let
        u = vector([1.0, 2.0, 3.0, 4.0])
        v = vector([1.0, 2.0, 3.0, 4.0])
        w = vector([1.0, 2.0, 2.999999999, 4.00000001])
        z = vector([1.0, 3.0, 3.0, 4.0])
      check u =~ v
      check v =~ w
      check v != w
      check w !=~ z
    test "approximate 32-bit vector equality":
      let
        u = vector([1'f32, 2'f32, 3'f32, 4'f32])
        v = vector([1'f32, 2'f32, 3'f32, 4'f32])
        w = vector([1'f32, 2'f32, 2.999999'f32, 4.000001'f32])
        z = vector([1'f32, 3'f32, 3'f32, 4'f32])
      check u =~ v
      check v =~ w
      check v != w
      check w !=~ z
    test "approximate matrix equality":
      let
        m = makeMatrix(3, 5, proc(i, j: int): float64 = (i + 3 * j).float64)
        n = makeMatrix(3, 5, proc(i, j: int): float64 = (i + 3 * j).float64)
        q = makeMatrix(3, 5, proc(i, j: int): float64 = (i - 2 * j).float64)
      var p = makeMatrix(3, 5, proc(i, j: int): float64 = (i + 3 * j).float64)
      p[2, 2] = p[2, 2] - 0.000000001
      p[1, 3] = p[1, 3] + 0.000000001
      check m =~ n
      check n =~ p
      check n != p
      check p !=~ q
    test "approximate 32-bit matrix equality":
      let
        m = makeMatrix(3, 5, proc(i, j: int): float32 = (i + 3 * j).float32)
        n = makeMatrix(3, 5, proc(i, j: int): float32 = (i + 3 * j).float32)
        q = makeMatrix(3, 5, proc(i, j: int): float32 = (i - 2 * j).float32)
      var p = makeMatrix(3, 5, proc(i, j: int): float32 = (i + 3 * j).float32)
      p[2, 2] = p[2, 2] - 0.000001
      p[1, 3] = p[1, 3] + 0.000001
      check m =~ n
      check n =~ p
      check n != p
      check p !=~ q

run()