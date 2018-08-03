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
  suite "trivial operations CUDA objects":
    test "reshape of matrices":
      let
        m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).gpu()
        m2 = matrix(@[
          @[1.0, 1.0, 2.0],
          @[-1.0, 2.0, -1.0],
          @[3.0, 2.0, 1.0],
          @[0.0, 3.0, 4.0]
        ]).gpu()
      check m1.reshape(4, 3) == m2
    test "turn vectors into matrices":
      let
        v = vector([1.0, -1.0, 3.0, 0.0, 1.0, 2.0, 2.0, 3.0, 2.0, -1.0, 1.0, 4.0]).gpu()
        m = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).gpu()
      check v.asMatrix(3, 4) == m
    test "turn matrices into vectors":
      let
        v = vector([1.0, -1.0, 3.0, 0.0, 1.0, 2.0, 2.0, 3.0, 2.0, -1.0, 1.0, 4.0]).gpu()
        m = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).gpu()
      check m.asVector == v
    test "hard transpose of matrices":
      var
        m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).gpu()
        m2 = matrix(@[
          @[1.0, -1.0, 3.0],
          @[0.0, 1.0, 2.0],
          @[2.0, 3.0, 2.0],
          @[-1.0, 1.0, 4.0]
        ]).gpu()

      check(m1.T == m2)

run()