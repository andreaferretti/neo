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

import unittest, neo, neo/statics

proc run() =
  suite "mixed slice assignments":
    test "assigning to a slice":
      var m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
      let n = statics.matrix([
          [5, 6, 7],
          [8, 9, 10],
          [11, 12, 13]
        ], rowMajor)
      m[1 .. 3, 1 .. 3] = n
      check m[2, 2] == 9
      check m[3, 2] == 12

    test "assigning a slice to another slice":
      var m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
      let n = statics.makeMatrixIJ(int, 5, 5, 2 * i + 2 * j)
      m[1 .. 3, 1 .. 3] = n[2 .. 4, 2 .. 4]
      check m[2, 2] == 12
      check m[3, 2] == 14

    test "assigning to a slice on columns only":
      var m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
      let n = statics.makeMatrixIJ(int, 5, 3, 2 * i + 2 * j, rowMajor)

      m[All, 2 .. 4] = n
      check m[2, 2] == 4
      check m[3, 2] == 6

    test "assigning to a slice on rows only":
      var m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
      let n = statics.makeMatrixIJ(int, 3, 5, 2 * i + 2 * j)

      m[2 .. 4, All] = n
      check m[2, 2] == 4
      check m[3, 2] == 6

    test "assigning to a slice with BLAS operations":
      var m = statics.makeMatrixIJ(float64, 5, 5, (3 * i + j).float64, rowMajor)
      let n = statics.matrix([
          [5'f64, 6, 7],
          [8'f64, 9, 10],
          [11'f64, 12, 13]
        ])
      m[1 .. 3, 1 .. 3] = n
      check m[2, 2] == 9'f64
      check m[3, 2] == 12'f64

  suite "mixed matrix operations":
    test "mixed matrix sum":
      let
        m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ], order = rowMajor)
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ])
        m3 = matrix([
          [4.0, 1.0, 1.0, 0.0],
          [1.0, 2.0, 0.0, 1.0],
          [7.0, 3.0, 4.0, 6.0]
        ])
      check(m1 + m2 == m3)
    test "mixed mutating matrix sum":
      var m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ], order = rowMajor)
      let
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ])
        m3 = matrix([
          [4.0, 1.0, 1.0, 0.0],
          [1.0, 2.0, 0.0, 1.0],
          [7.0, 3.0, 4.0, 6.0]
        ])
      m1 += m2
      check m1 == m3
    test "mixed matrix difference":
      let
        m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ])
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ], order = rowMajor)
        m3 = matrix([
          [-2.0, -1.0, 3.0, -2.0],
          [-3.0, 0.0, 6.0, 1.0],
          [-1.0, 1.0, 0.0, 2.0]
        ])
      check(m1 - m2 == m3)
    test "mutating matrix sum":
      var m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ], order = rowMajor)
      let
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ])
        m3 = matrix([
          [-2.0, -1.0, 3.0, -2.0],
          [-3.0, 0.0, 6.0, 1.0],
          [-1.0, 1.0, 0.0, 2.0]
        ])
      m1 -= m2
      check m1 == m3
    test "mixed matrix multiplication":
      let
        m1 = matrix([
          [1.0, 1.0, 2.0, -3.0],
          [3.0, 0.0, -7.0, 2.0]
        ], order = rowMajor)
        m2 = matrix([
          [1.0, 1.0, 2.0],
          [3.0, 1.0, -5.0],
          [-1.0, -1.0, 2.0],
          [4.0, 2.0, 3.0]
        ])
        m3 = matrix([
          [-10.0, -6.0, -8.0],
          [18.0, 14.0, -2.0]
        ])
      check(m1 * m2 == m3)
    test "mixed matrix multiplication take two":
      let
        m1 = matrix([
          [1.0, 1.0, 2.0, -3.0],
          [3.0, 0.0, -7.0, 2.0]
        ])
        m2 = matrix([
          [1.0, 1.0, 2.0],
          [3.0, 1.0, -5.0],
          [-1.0, -1.0, 2.0],
          [4.0, 2.0, 3.0]
        ], order = rowMajor)
        m3 = matrix([
          [-10.0, -6.0, -8.0],
          [18.0, 14.0, -2.0]
        ])
      check(m1 * m2 == m3)
    test "mixed matrix Hadamard multiplication":
      let
        m1 = matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ])
        m2 = matrix([
          [3.0, 1.0, -1.0, 1.0],
          [2.0, 1.0, -3.0, 0.0],
          [4.0, 1.0, 2.0, 2.0]
        ], order = rowMajor)
        m3 = matrix([
          [3.0, 0.0, -2.0, -1.0],
          [-2.0, 1.0, -9.0, 0.0],
          [12.0, 2.0, 4.0, 8.0]
        ])
      check((m1 |*| m2) == m3)

run()