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


suite "cloning and slicing":
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
  test "slicing a vector":
    let
      v1 = vector([1'f64, 2, 3, 4, 5, 6, 7, 8, 9])
      v2 = v1.gpu()
      v3 = v2[3 .. 6]
      v4 = v3.cpu()
    check v4 == vector([4'f64, 5, 6, 7])
  test "slicing a matrix":
    let
      m1 = matrix(@[
        @[1'f64, 2, 3, 4, 5],
        @[3'f64, 1, 1, 6, 8],
        @[0'f64, 8, 2, 7, 0]
      ])
      m2 = m1.gpu()
      m3 = m2[1 .. 3]
      m4 = m3.cpu()
      m5 = matrix(@[
        @[2'f64, 3, 4],
        @[1'f64, 1, 6],
        @[8'f64, 2, 7]
      ])
    check m4 == m5