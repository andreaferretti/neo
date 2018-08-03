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
  suite "cloning":
    test "cloning a vector":
      let
        v1 = randomVector(10)
        v2 = v1.gpu()
        v3 = v2.clone()
        v4 = v3.cpu()
      check v1 == v4
    test "cloning a matrix":
      let
        m1 = randomMatrix(10, 7)
        m2 = m1.gpu()
        m3 = m2.clone()
        m4 = m3.cpu()
      check m1 == m4

run()