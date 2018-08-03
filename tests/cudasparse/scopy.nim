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

import unittest, neo/sparse, neo/cudasparse

proc run() =
  suite "copying back and forth":
    test "copy of a 64-bit vector":
      let
        v1 = sparseVector(10, @[3'i32, 5, 7], @[2.0, 3, -1])
        v2 = v1.gpu()
        v3 = v2.cpu()
      check v1 == v3
    test "copy of a 64-bit matrix":
      let
        m1 = csr(
          rows = @[0'i32, 3, 4, 7, 9],
          cols = @[0'i32, 2, 3, 1, 0, 2, 3, 1, 3],
          vals = @[1'f64, 2, 3, 4, 5, 6, 7, 8, 9],
          numCols = 4
        )
        m2 = m1.gpu()
        m3 = m2.cpu()
      check m1 == m3

run()