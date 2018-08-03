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

import unittest, neo/dense, neo/cudadense

proc run() =
  suite "vector and matrix equality":
    test "strict 32-bit vector equality":
      let
        u = vector([1'f32, 2'f32, 3'f32, 4'f32]).gpu()
        v = vector([1'f32, 2'f32, 3'f32, 4'f32]).gpu()
        w = vector([1'f32, 3'f32, 3'f32, 4'f32]).gpu()
      check u == v
      check v != w
    test "approximate 32-bit vector equality":
      let
        u = vector([1'f32, 2'f32, 3'f32, 4'f32]).gpu()
        v = vector([1'f32, 2'f32, 3'f32, 4'f32]).gpu()
        w = vector([1'f32, 2'f32, 2.999999'f32, 4.000001'f32]).gpu()
        z = vector([1'f32, 3'f32, 3'f32, 4'f32]).gpu()
      check u =~ v
      check v =~ w
      check v != w
      check w !=~ z
    test "strict 32-bit matrix equality":
      let
        m = makeMatrix(3, 5, proc(i, j: int): float32 = (i + 3 * j).float32).gpu()
        n = makeMatrix(3, 5, proc(i, j: int): float32 = (i + 3 * j).float32).gpu()
        p = makeMatrix(3, 5, proc(i, j: int): float32 = (i - 2 * j).float32).gpu()
      check m == n
      check n != p
    test "strict 64-bit vector equality":
      let
        u = vector([1.0, 2.0, 3.0, 4.0]).gpu()
        v = vector([1.0, 2.0, 3.0, 4.0]).gpu()
        w = vector([1.0, 3.0, 3.0, 4.0]).gpu()
      check u == v
      check v != w
    test "approximate 64-bit vector equality":
      let
        u = vector([1.0, 2.0, 3.0, 4.0]).gpu()
        v = vector([1.0, 2.0, 3.0, 4.0]).gpu()
        w = vector([1.0, 2.0, 2.999999, 4.000001]).gpu()
        z = vector([1.0, 3.0, 3.0, 4.0]).gpu()
      check u =~ v
      check v =~ w
      check v != w
      check w !=~ z
    test "strict 64-bit matrix equality":
      let
        m = makeMatrix(3, 5, proc(i, j: int): float64 = (i + 3 * j).float64).gpu()
        n = makeMatrix(3, 5, proc(i, j: int): float64 = (i + 3 * j).float64).gpu()
        p = makeMatrix(3, 5, proc(i, j: int): float64 = (i - 2 * j).float64).gpu()
      check m == n
      check n != p

run()