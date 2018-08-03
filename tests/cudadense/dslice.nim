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
  suite "slicing CUDA vectors":
    test "getting a slice of a vector":
      let
        v = vector(@[1'f64, 2, 3, 4, 5]).gpu()
        w = v[2 .. 3]

      check w.cpu() == vector(3'f64, 4'f64)

    test "assigning to a slice":
      var v = vector(@[1'f64, 2, 3, 4, 5]).gpu()
      let w = vector(6'f64, 7'f64).gpu()

      v[2 .. 3] = w
      check v.cpu() == vector(@[1'f64, 2, 6, 7, 5])

    test "assigning a slice to another slice":
      var v = vector(@[1'f64, 2, 3, 4, 5]).gpu()
      let w = vector(@[6'f64, 7, 8, 9, 10]).gpu()

      v[2 .. 3] = w[3 .. 4]
      check v.cpu() == vector(@[1'f64, 2, 9, 10, 5])

  suite "slicing CUDA matrices":
    test "slice of a full matrix":
      let
        m = makeMatrixIJ(int, 5, 5, 3 * i + j).gpu()
        s = m[1 .. 3, 2 .. 4]
        expected = matrix(@[
          @[5, 6, 7],
          @[8, 9, 10],
          @[11, 12, 13]
        ]).gpu()

      check s == expected
    test "slice on columns only":
      let
        m = makeMatrixIJ(int, 5, 5, 3 * i + j).gpu()
        s = m[All, 2 .. 4]
        expected = matrix(@[
          @[2, 3, 4],
          @[5, 6, 7],
          @[8, 9, 10],
          @[11, 12, 13],
          @[14, 15, 16]
        ]).gpu()

      check s == expected
    test "slice on rows only":
      let
        m = makeMatrixIJ(int, 5, 5, 3 * i + j).gpu()
        s = m[1 .. 3, All]
        expected = matrix(@[
          @[3, 4, 5, 6, 7],
          @[6, 7, 8, 9, 10],
          @[9, 10, 11, 12, 13],
        ]).gpu()

      check s == expected
    test "slice a sliced matrix":
      let
        m = makeMatrixIJ(int, 5, 5, 3 * i + j).gpu()
        s1 = m[1 .. 4, 1 .. 4]
        s2 = s1[0 .. 2, 1 .. 3]
        expected = matrix(@[
          @[5, 6, 7],
          @[8, 9, 10],
          @[11, 12, 13]
        ]).gpu()

      check s2 == expected
    test "slice a sliced matrix on rows only":
      let
        m = makeMatrixIJ(int, 5, 5, 3 * i + j).gpu()
        s1 = m[1 .. 4, 1 .. 4]
        s2 = s1[0 .. 2, All]
        expected = matrix(@[
          @[4, 5, 6, 7],
          @[7, 8, 9, 10],
          @[10, 11, 12, 13]
        ]).gpu()

      check s2 == expected
    test "assigning to a slice":
      var m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64).gpu()
      let n = matrix(@[
          @[5'f64, 6, 7],
          @[8'f64, 9, 10],
          @[11'f64, 12, 13]
        ]).gpu()
      m[1 .. 3, 1 .. 3] = n
      let m1 = m.cpu()
      check m1[2, 2] == 9'f64
      check m1[3, 2] == 12'f64
    test "rows of a slice of a matrix":
      let
        m = makeMatrixIJ(int, 5, 5, 3 * i + j).gpu()
        s = m[1 .. 3, 2 .. 4]
        r = s.row(1)

      check r == vector(8, 9, 10).gpu()
    test "columns of a slice of a matrix":
      let
        m = makeMatrixIJ(int, 5, 5, 3 * i + j).gpu()
        s = m[1 .. 3, 2 .. 4]
        r = s.column(1)

      check r == vector(6, 9, 12).gpu()
    test "rows of a slice of a sliced matrix":
      let
        m = makeMatrixIJ(int, 5, 5, 3 * i + j).gpu()
        s1 = m[1 .. 4, 1 .. 4]
        s2 = s1[0 .. 2, 1 .. 3]
        r = s2.row(1)

      check r == vector(8, 9, 10).gpu()
    test "matrix/vector multiplication on slices":
      let
        m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64).gpu()
        s = m[1 .. 3, 2 .. 4]
        n = matrix(@[
          @[5.0, 6, 7],
          @[8.0, 9, 10],
          @[11.0, 12, 13]
        ]).gpu()
        v = n.column(2)

      check(s * v == n * v)
    test "matrix multiplication on slices":
      let
        m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64).gpu()
        s = m[1 .. 3, 2 .. 4]
        n = matrix(@[
          @[5.0, 6, 7],
          @[8.0, 9, 10],
          @[11.0, 12, 13]
        ]).gpu()

      check(s * s == n * n)
    test "matrix addition on slices":
      let
        m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64).gpu()
        s = m[1 .. 3, 2 .. 4]
        n = matrix(@[
          @[5.0, 6, 7],
          @[8.0, 9, 10],
          @[11.0, 12, 13]
        ]).gpu()

      check(s + s == n + n)
    test "matrix subtraction on slices":
      let
        m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64).gpu()
        s = m[1 .. 3, 2 .. 4]
        n = matrix(@[
          @[5.0, 6, 7],
          @[8.0, 9, 10],
          @[11.0, 12, 13]
        ]).gpu()

      check(s - n == zeros(3, 3).gpu())
    test "scaling slices":
      var
        m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64).gpu()
        s = m[1 .. 3, 2 .. 4]
      s *= 2
      let expected = matrix(@[
        @[0.0,  1.0,  2.0,  3.0,  4.0],
        @[3.0,  4.0,  10.0, 12.0, 14.0],
        @[6.0,  7.0,  16.0, 18.0, 20.0],
        @[9.0,  10.0, 22.0, 24.0, 26.0],
        @[12.0, 13.0, 14.0, 15.0, 16.0]
      ]).gpu()

      check(m == expected)
    test "mutable sum on slices":
      var
        m = makeMatrixIJ(float64, 5, 5, (3 * i + j).float64).gpu()
        s = m[1 .. 3, 2 .. 4]
        t = makeMatrixIJ(float64, 3, 3, (i - j).float64).gpu()
      s += t
      let expected = matrix(@[
        @[0.0,  1.0,  2.0,  3.0,  4.0],
        @[3.0,  4.0,  5.0,  5.0,  5.0],
        @[6.0,  7.0,  9.0,  9.0,  9.0],
        @[9.0,  10.0, 13.0, 13.0, 13.0],
        @[12.0, 13.0, 14.0, 15.0, 16.0]
      ]).gpu()

      check(m == expected)

run()