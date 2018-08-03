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
  suite "iterators on matrices":
    test "rows matrix iterator":
      let m = makeMatrix(3, 2, proc(i, j: int): float64 = (i + 2 * j + 1).float64).gpu()
      var
        sum = zeros(2).gpu()
        count = 0
      for r in m.rows:
        sum += r
        count += 1
      check sum == vector(6.0, 12.0).gpu()
      check count == 3
    test "columns matrix iterator":
      let m = makeMatrix(3, 2, proc(i, j: int): float64 = (i + 2 * j + 1).float64).gpu()
      var
        sum = zeros(3).gpu()
        count = 0
      for c in m.columns:
        sum += c
        count += 1
      check sum == vector(4.0, 6.0, 8.0).gpu()
      check count == 2
    test "rows matrix iterator":
      let m = makeMatrix(3, 2, proc(i, j: int): float64 = (i + 2 * j + 1).float64).gpu()
      var
        sum = zeros(2).gpu()
        count = 0
      for r in m.rowsSlow:
        sum += r
        count += 1
      check sum == vector(6.0, 12.0).gpu()
      check count == 3
    test "columns matrix iterator":
      let m = makeMatrix(3, 2, proc(i, j: int): float64 = (i + 2 * j + 1).float64).gpu()
      var
        sum = zeros(3).gpu()
        count = 0
      for c in m.columnsSlow:
        sum += c
        count += 1
      check sum == vector(4.0, 6.0, 8.0).gpu()
      check count == 2

run()