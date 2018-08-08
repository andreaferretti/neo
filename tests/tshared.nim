# Copyright 2018 UniCredit S.p.A.
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

import unittest, threadpool, neo/dense

type MultiplyData = object
  m: Matrix[float64]
  firstRow, lastRow: int
  k: float64

proc multiplyMatrix(data: MultiplyData) =
  var m = data.m[data.firstRow .. data.lastRow, All]
  m *= data.k

proc run() =
  suite "parallel operation on shared matrices":
    test "multiplying a shared matrix in parallel":
      var m = sharedMatrix(100, 100, float64)
      defer:
        dealloc(m)

      for i in 0 .. 99:
        for j in 0 .. 99:
          m[i, j] = (i + j).float64

      check m[20, 10] == 30.0
      check m[70, 10] == 80.0

      spawn multiplyMatrix(MultiplyData(m: m, firstRow: 0, lastRow: 49, k: 2))
      spawn multiplyMatrix(MultiplyData(m: m, firstRow: 50, lastRow: 99, k: 3))
      sync()

      check m[20, 10] == 60.0
      check m[70, 10] == 240.0

run()