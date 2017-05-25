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

suite "iterators on vectors":
  test "items vector iterator":
    let v = @[1.0, 3.0, 2.0, 8.0, -2.0]
    var
      sum = 0.0
      count = 0
    for x in v:
      sum += x
      count += 1
    check sum == 12.0
    check count == 5
  test "pairs vector iterator":
    let v = @[1.0, 3.0, 2.0, 8.0, -2.0]
    var
      sum = 0.0
      sumI = 0
    for i, x in v:
      sum += x
      sumI += i
    check sum == 12.0
    check sumI == 10

suite "iterators on 32-bit vectors":
  test "items vector iterator":
    let v = @[1'f32, 3'f32, 2'f32, 8'f32, -2'f32]
    var
      sum = 0'f32
      count = 0
    for x in v:
      sum += x
      count += 1
    check sum == 12'f32
    check count == 5
  test "pairs vector iterator":
    let v = @[1'f32, 3'f32, 2'f32, 8'f32, -2'f32]
    var
      sum = 0'f32
      sumI = 0
    for i, x in v:
      sum += x
      sumI += i
    check sum == 12'f32
    check sumI == 10

suite "iterators on matrices":
  test "items matrix iterator":
    let m = makeMatrix(3, 2, proc(i, j: int): float64 = (i + 2 * j + 1).float64)
    var
      sum = 0.0
      count = 0
    for x in m:
      sum += x
      count += 1
    check sum == 18.0
    check count == 6
  test "pairs matrix iterator":
    let m = makeMatrix(3, 2, proc(i, j: int): float64 = (i + 2 * j + 1).float64)
    var
      sum = 0.0
      sumI = 0
    for t, x in m:
      let (i, j) = t
      sum += x
      sumI += (i + j)
    check sum == 18.0
    check sumI == 9
  test "rows matrix iterator":
    let m = makeMatrix(3, 2, proc(i, j: int): float64 = (i + 2 * j + 1).float64)
    var
      sum = 0.0
      count = 0
    for r in m.rows:
      sum += (r[0] + r[1])
      count += 1
    check sum == 18.0
    check count == 3
  test "columns matrix iterator":
    let m = makeMatrix(3, 2, proc(i, j: int): float64 = (i + 2 * j + 1).float64)
    var
      sum = 0.0
      count = 0
    for c in m.columns:
      sum += (c[0] + c[1] + c[2])
      count += 1
    check sum == 18.0
    check count == 2