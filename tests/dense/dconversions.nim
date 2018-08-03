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
  suite "precision conversions":
    test "vectors: 64 to 32 bits":
      let v = vector([1.0, 3.5, 2.0, 4.5])
      check v.to32 == vector([1'f32, 3.5'f32, 2'f32, 4.5'f32])
    test "vectors: 32 to 64 bits":
      let v = vector([1'f32, 3.5'f32, 2'f32, 4.5'f32])
      check v.to64 == vector([1.0, 3.5, 2.0, 4.5])
    test "matrices: 64 to 32 bits":
      let
        m = makeMatrix(3, 5, proc(i, j: int): float64 = (i + j).float64)
        n = makeMatrix(3, 5, proc(i, j: int): float32 = (i + j).float32)
      check m.to32 == n
    test "matrices: 32 to 64 bits":
      let
        m = makeMatrix(3, 5, proc(i, j: int): float64 = (i + j).float64)
        n = makeMatrix(3, 5, proc(i, j: int): float32 = (i + j).float32)
      check n.to64 == m

run()