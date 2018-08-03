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
  suite "universal functions":
    test "universal sqrt on vectors":
      let u = vector([1.0, 4.0, 9.0, 16.0])
      check sqrt(u) == vector([1.0, 2.0, 3.0, 4.0])
    test "universal sine on matrices":
      let m = matrix(@[@[1.0, 2.0], @[4.0, 8.0]])
      check sin(m) == matrix(@[@[sin(1.0), sin(2.0)], @[sin(4.0), sin(8.0)]])
    test "defining a new universal function":
      proc plusFive(x: float64): float64 = x + 5
      makeUniversalLocal(plusFive)
      let v = vector([1.0, 4.0, 9.0, 16.0])
      check plusFive(v) == vector([6.0, 9.0, 14.0, 21.0])


  suite "32-bit universal functions":
    test "universal sqrt on vectors":
      let u = vector([1'f32, 4'f32, 9'f32, 16'f32])
      check sqrt(u) == vector([1'f32, 2'f32, 3'f32, 4'f32])
    test "universal sine on matrices":
      let m = matrix(@[@[1'f32, 2'f32], @[4'f32, 8'f32]])
      check sin(m) == matrix(@[@[sin(1'f32), sin(2'f32)], @[sin(4'f32), sin(8'f32)]])
    test "defining a new universal function":
      proc plusFive(x: float64): float64 = x + 5
      makeUniversalLocal(plusFive)
      let v = vector([1'f32, 4'f32, 9'f32, 16'f32])
      check plusFive(v) == vector([6'f32, 9'f32, 14'f32, 21'f32])

run()