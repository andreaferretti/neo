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

import unittest, nimat/dense

suite "mixed matrix operations":
  test "mixed matrix sum":
    let
      m1 = matrix(@[
        @[1.0, 0.0, 2.0, -1.0],
        @[-1.0, 1.0, 3.0, 1.0],
        @[3.0, 2.0, 2.0, 4.0]
      ], order = rowMajor)
      m2 = matrix(@[
        @[3.0, 1.0, -1.0, 1.0],
        @[2.0, 1.0, -3.0, 0.0],
        @[4.0, 1.0, 2.0, 2.0]
      ])
      m3 = matrix(@[
        @[4.0, 1.0, 1.0, 0.0],
        @[1.0, 2.0, 0.0, 1.0],
        @[7.0, 3.0, 4.0, 6.0]
      ])
    check(m1 + m2 == m3)
  test "mixed mutating matrix sum":
    var m1 = matrix(@[
        @[1.0, 0.0, 2.0, -1.0],
        @[-1.0, 1.0, 3.0, 1.0],
        @[3.0, 2.0, 2.0, 4.0]
      ], order = rowMajor)
    let
      m2 = matrix(@[
        @[3.0, 1.0, -1.0, 1.0],
        @[2.0, 1.0, -3.0, 0.0],
        @[4.0, 1.0, 2.0, 2.0]
      ])
      m3 = matrix(@[
        @[4.0, 1.0, 1.0, 0.0],
        @[1.0, 2.0, 0.0, 1.0],
        @[7.0, 3.0, 4.0, 6.0]
      ])
    m1 += m2
    check m1 == m3
  test "mixed matrix difference":
    let
      m1 = matrix(@[
        @[1.0, 0.0, 2.0, -1.0],
        @[-1.0, 1.0, 3.0, 1.0],
        @[3.0, 2.0, 2.0, 4.0]
      ])
      m2 = matrix(@[
        @[3.0, 1.0, -1.0, 1.0],
        @[2.0, 1.0, -3.0, 0.0],
        @[4.0, 1.0, 2.0, 2.0]
      ], order = rowMajor)
      m3 = matrix(@[
        @[-2.0, -1.0, 3.0, -2.0],
        @[-3.0, 0.0, 6.0, 1.0],
        @[-1.0, 1.0, 0.0, 2.0]
      ])
    check(m1 - m2 == m3)
  test "mutating matrix sum":
    var m1 = matrix(@[
        @[1.0, 0.0, 2.0, -1.0],
        @[-1.0, 1.0, 3.0, 1.0],
        @[3.0, 2.0, 2.0, 4.0]
      ], order = rowMajor)
    let
      m2 = matrix(@[
        @[3.0, 1.0, -1.0, 1.0],
        @[2.0, 1.0, -3.0, 0.0],
        @[4.0, 1.0, 2.0, 2.0]
      ])
      m3 = matrix(@[
        @[-2.0, -1.0, 3.0, -2.0],
        @[-3.0, 0.0, 6.0, 1.0],
        @[-1.0, 1.0, 0.0, 2.0]
      ])
    m1 -= m2
    check m1 == m3
  test "mixed matrix multiplication":
    let
      m1 = matrix(@[
        @[1.0, 1.0, 2.0, -3.0],
        @[3.0, 0.0, -7.0, 2.0]
      ], order = rowMajor)
      m2 = matrix(@[
        @[1.0, 1.0, 2.0],
        @[3.0, 1.0, -5.0],
        @[-1.0, -1.0, 2.0],
        @[4.0, 2.0, 3.0]
      ])
      m3 = matrix(@[
        @[-10.0, -6.0, -8.0],
        @[18.0, 14.0, -2.0]
      ])
    check(m1 * m2 == m3)
  test "mixed matrix multiplication take two":
    let
      m1 = matrix(@[
        @[1.0, 1.0, 2.0, -3.0],
        @[3.0, 0.0, -7.0, 2.0]
      ])
      m2 = matrix(@[
        @[1.0, 1.0, 2.0],
        @[3.0, 1.0, -5.0],
        @[-1.0, -1.0, 2.0],
        @[4.0, 2.0, 3.0]
      ], order = rowMajor)
      m3 = matrix(@[
        @[-10.0, -6.0, -8.0],
        @[18.0, 14.0, -2.0]
      ])
    check(m1 * m2 == m3)
  test "mixed matrix Hadamard multiplication":
    let
      m1 = matrix(@[
        @[1.0, 0.0, 2.0, -1.0],
        @[-1.0, 1.0, 3.0, 1.0],
        @[3.0, 2.0, 2.0, 4.0]
      ])
      m2 = matrix(@[
        @[3.0, 1.0, -1.0, 1.0],
        @[2.0, 1.0, -3.0, 0.0],
        @[4.0, 1.0, 2.0, 2.0]
      ], order = rowMajor)
      m3 = matrix(@[
        @[3.0, 0.0, -2.0, -1.0],
        @[-2.0, 1.0, -9.0, 0.0],
        @[12.0, 2.0, 4.0, 8.0]
      ])
    check((m1 |*| m2) == m3)