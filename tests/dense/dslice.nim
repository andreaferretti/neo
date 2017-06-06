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


suite "slicing matrices":
  test "slice of a full matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j)
      s = m[1 .. 3, 2 .. 4]
      expected = matrix(@[
        @[5, 6, 7],
        @[8, 9, 10],
        @[11, 12, 13]
      ])

    check s == expected
  test "slice on columns only":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j)
      s = m[All, 2 .. 4]
      expected = matrix(@[
        @[2, 3, 4],
        @[5, 6, 7],
        @[8, 9, 10],
        @[11, 12, 13],
        @[14, 15, 16]
      ])

    check s == expected
  test "slice on rows only":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j)
      s = m[1 .. 3, All]
      expected = matrix(@[
        @[3, 4, 5, 6, 7],
        @[6, 7, 8, 9, 10],
        @[9, 10, 11, 12, 13],
      ])

    check s == expected
  test "slice a sliced matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j)
      s1 = m[1 .. 4, 1 .. 4]
      s2 = s1[0 .. 2, 1 .. 3]
      expected = matrix(@[
        @[5, 6, 7],
        @[8, 9, 10],
        @[11, 12, 13]
      ])


    check s2 == expected
  test "slice a sliced matrix on rows only":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j)
      s1 = m[1 .. 4, 1 .. 4]
      s2 = s1[0 .. 2, All]
      expected = matrix(@[
        @[4, 5, 6, 7],
        @[7, 8, 9, 10],
        @[10, 11, 12, 13]
      ])


    check s2 == expected