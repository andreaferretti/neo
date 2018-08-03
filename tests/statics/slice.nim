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
  suite "slicing vectors":
    test "getting a slice of a vector":
      let
        v = statics.vector([1, 2, 3, 4, 5])
        w = v[2 .. 3]

      check w == statics.vector([3, 4])

    test "assigning to a slice":
      var v = statics.vector([1, 2, 3, 4, 5])
      let w = statics.vector([6, 7])

      v[2 .. 3] = w
      check v == statics.vector([1, 2, 6, 7, 5])

    test "assigning to a slice with BLAS operations":
      var v = statics.vector([1.0, 2.0, 3.0, 4.0, 5.0])
      let
        w = statics.vector([6.0, 7.0])
        expected = statics.vector([1.0, 2.0, 6.0, 7.0, 5.0])

      v[2 .. 3] = w
      check v == expected

    test "assigning a slice to another slice":
      var v = statics.vector([1, 2, 3, 4, 5])
      let w = statics.vector([6, 7, 8, 9, 10])

      v[2 .. 3] = w[3 .. 4]
      check v == statics.vector([1, 2, 9, 10, 5])

  suite "slicing column major matrices":
    test "slice of a full matrix":
      let
        m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
        s = m[1 .. 3, 2 .. 4]
        expected = statics.matrix([
          [5, 6, 7],
          [8, 9, 10],
          [11, 12, 13]
        ])

      check s == expected
    test "slice on columns only":
      let
        m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
        s = m[All, 2 .. 4]
        expected = matrix([
          [2, 3, 4],
          [5, 6, 7],
          [8, 9, 10],
          [11, 12, 13],
          [14, 15, 16]
        ])

      check s == expected
    test "slice on rows only":
      let
        m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
        s = m[1 .. 3, All]
        expected = statics.matrix([
          [3, 4, 5, 6, 7],
          [6, 7, 8, 9, 10],
          [9, 10, 11, 12, 13],
        ])

      check s == expected
    test "slice a sliced matrix":
      let
        m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
        s1 = m[1 .. 4, 1 .. 4]
        s2 = s1[0 .. 2, 1 .. 3]
        expected = statics.matrix([
          [5, 6, 7],
          [8, 9, 10],
          [11, 12, 13]
        ])

      check s2 == expected
    test "slice a sliced matrix on rows only":
      let
        m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
        s1 = m[1 .. 4, 1 .. 4]
        s2 = s1[0 .. 2, All]
        expected = statics.matrix([
          [4, 5, 6, 7],
          [7, 8, 9, 10],
          [10, 11, 12, 13]
        ])

      check s2 == expected

    test "assigning to a slice":
      var m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
      let n = statics.matrix([
          [5, 6, 7],
          [8, 9, 10],
          [11, 12, 13]
        ])
      m[1 .. 3, 1 .. 3] = n
      check m[2, 2] == 9
      check m[3, 2] == 12

    test "assigning a slice to another slice":
      var m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
      let n = statics.makeMatrixIJ(int, 5, 5, 2 * i + 2 * j)
      m[1 .. 3, 1 .. 3] = n[2 .. 4, 2 .. 4]
      check m[2, 2] == 12
      check m[3, 2] == 14

    test "assigning to a slice on columns only":
      var m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
      let n = statics.makeMatrixIJ(int, 5, 3, 2 * i + 2 * j)

      m[All, 2 .. 4] = n
      check m[2, 2] == 4
      check m[3, 2] == 6

    test "assigning to a slice on rows only":
      var m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
      let n = statics.makeMatrixIJ(int, 3, 5, 2 * i + 2 * j)

      m[2 .. 4, All] = n
      check m[2, 2] == 4
      check m[3, 2] == 6

    test "assigning to a slice with BLAS operations":
      var m = statics.makeMatrixIJ(float64, 5, 5, (3 * i + j).float64)
      let n = statics.matrix([
          [5'f64, 6, 7],
          [8'f64, 9, 10],
          [11'f64, 12, 13]
        ])
      m[1 .. 3, 1 .. 3] = n
      check m[2, 2] == 9'f64
      check m[3, 2] == 12'f64

    test "slice of a matrix should share storage":
      var
        m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
        s = m[1 .. 3, 2 .. 4]

      s[1, 1] = 0

      check m[2, 3] == 0
    test "rows of a slice of a matrix":
      let
        m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
        s = m[1 .. 3, 2 .. 4]
        r = s.row(1)

      check r == statics.vector([8, 9, 10])
    test "columns of a slice of a matrix":
      let
        m = statics.makeMatrixIJ(int, 5, 5, 3 * i + j)
        s = m[1 .. 3, 2 .. 4]
        r = s.column(1)

      check r == statics.vector([6, 9, 12])

run()