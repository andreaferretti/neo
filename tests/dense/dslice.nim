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


suite "slicing column major matrices":
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
  test "slice of a matrix should share storage":
    var
      m = makeMatrixIJ(int, 5, 5, 3 * i + j)
      s = m[1 .. 3, 2 .. 4]

    s[1, 1] = 0

    check m[2, 3] == 0
  test "rows of a slice of a matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j)
      s = m[1 .. 3, 2 .. 4]
      r = s.row(1)

    check r == vector(8, 9, 10)
  test "columns of a slice of a matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j)
      s = m[1 .. 3, 2 .. 4]
      r = s.column(1)

    check r == vector(6, 9, 12)
  test "rows of a slice of a sliced matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j)
      s1 = m[1 .. 4, 1 .. 4]
      s2 = s1[0 .. 2, 1 .. 3]
      r = s2.row(1)

    check r == vector(8, 9, 10)
  test "matrix/vector multiplication on slices":
    let
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64)
      s = m[1 .. 3, 2 .. 4]
      n = matrix(@[
        @[5.0, 6, 7],
        @[8.0, 9, 10],
        @[11.0, 12, 13]
      ])
      v = n.column(2)

    check(s * v == n * v)
  test "matrix multiplication on slices":
    let
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64)
      s = m[1 .. 3, 2 .. 4]
      n = matrix(@[
        @[5.0, 6, 7],
        @[8.0, 9, 10],
        @[11.0, 12, 13]
      ])

    check(s * s == n * n)
  test "matrix addition on slices":
    let
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64)
      s = m[1 .. 3, 2 .. 4]
      n = matrix(@[
        @[5.0, 6, 7],
        @[8.0, 9, 10],
        @[11.0, 12, 13]
      ])

    check(s + s == n + n)
  test "matrix subtraction on slices":
    let
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64)
      s = m[1 .. 3, 2 .. 4]
      n = matrix(@[
        @[5.0, 6, 7],
        @[8.0, 9, 10],
        @[11.0, 12, 13]
      ])

    check(s - n == zeros(3, 3))
  test "scaling slices":
    var
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64)
      s = m[1 .. 3, 2 .. 4]
    s *= 2
    let expected = matrix(@[
      @[0.0,  1.0,  2.0,  3.0,  4.0],
      @[3.0,  4.0,  10.0, 12.0, 14.0],
      @[6.0,  7.0,  16.0, 18.0, 20.0],
      @[9.0,  10.0, 22.0, 24.0, 26.0],
      @[12.0, 13.0, 14.0, 15.0, 16.0]
    ])

    check(m == expected)

suite "slicing row major matrices":
  test "slice of a full matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
      s = m[1 .. 3, 2 .. 4]
      expected = matrix(@[
        @[5, 6, 7],
        @[8, 9, 10],
        @[11, 12, 13]
      ])

    check s == expected
  test "slice on columns only":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
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
      m = makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
      s = m[1 .. 3, All]
      expected = matrix(@[
        @[3, 4, 5, 6, 7],
        @[6, 7, 8, 9, 10],
        @[9, 10, 11, 12, 13],
      ])

    check s == expected
  test "slice a sliced matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
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
      m = makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
      s1 = m[1 .. 4, 1 .. 4]
      s2 = s1[0 .. 2, All]
      expected = matrix(@[
        @[4, 5, 6, 7],
        @[7, 8, 9, 10],
        @[10, 11, 12, 13]
      ])

    check s2 == expected
  test "slice of a matrix should share storage":
    var
      m = makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
      s = m[1 .. 3, 2 .. 4]

    s[1, 1] = 0

    check m[2, 3] == 0
  test "rows of a slice of a matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
      s = m[1 .. 3, 2 .. 4]
      r = s.row(1)

    check r == vector(8, 9, 10)
  test "columns of a slice of a matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
      s = m[1 .. 3, 2 .. 4]
      r = s.column(1)

    check r == vector(6, 9, 12)
  test "rows of a slice of a sliced matrix":
    let
      m = makeMatrixIJ(int, 5, 5, 3 * i + j, rowMajor)
      s1 = m[1 .. 4, 1 .. 4]
      s2 = s1[0 .. 2, 1 .. 3]
      r = s2.row(1)

    check r == vector(8, 9, 10)
  test "matrix/vector multiplication on slices":
    let
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64, rowMajor)
      s = m[1 .. 3, 2 .. 4]
      n = matrix(@[
        @[5.0, 6, 7],
        @[8.0, 9, 10],
        @[11.0, 12, 13]
      ])
      v = n.column(2)

    check(s * v == n * v)
  test "matrix multiplication on slices":
    let
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64, rowMajor)
      s = m[1 .. 3, 2 .. 4]
      n = matrix(@[
        @[5.0, 6, 7],
        @[8.0, 9, 10],
        @[11.0, 12, 13]
      ])

    check(s * s == n * n)
  test "matrix addition on slices":
    let
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64, rowMajor)
      s = m[1 .. 3, 2 .. 4]
      n = matrix(@[
        @[5.0, 6, 7],
        @[8.0, 9, 10],
        @[11.0, 12, 13]
      ])

    check(s + s == n + n)
  test "matrix subtraction on slices":
    let
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64, rowMajor)
      s = m[1 .. 3, 2 .. 4]
      n = matrix(@[
        @[5.0, 6, 7],
        @[8.0, 9, 10],
        @[11.0, 12, 13]
      ])

    check(s - n == zeros(3, 3))
  test "scaling slices":
    var
      m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64, rowMajor)
      s = m[1 .. 3, 2 .. 4]
    s *= 2
    let expected = matrix(@[
      @[0.0,  1.0,  2.0,  3.0,  4.0],
      @[3.0,  4.0,  10.0, 12.0, 14.0],
      @[6.0,  7.0,  16.0, 18.0, 20.0],
      @[9.0,  10.0, 22.0, 24.0, 26.0],
      @[12.0, 13.0, 14.0, 15.0, 16.0]
    ])

    check(m == expected)