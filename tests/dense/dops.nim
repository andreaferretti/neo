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

proc run() =
  suite "vector operations":
    test "scalar vector multiplication":
      let v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
      check((v * 2.0) == vector([2.0, 6.0, 4.0, 16.0, -4.0]))
      check((-1.0 * v) == vector([-1.0, -3.0, -2.0, -8.0, 2.0]))
    test "in place scalar vector multiplication":
      var v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
      v *= 2.0
      check v == vector([2.0, 6.0, 4.0, 16.0, -4.0])
    test "scalar vector division":
      let v = vector([2.0, 6.0, 4.0, 16.0, -4.0])
      check((v / 2.0) == vector([1.0, 3.0, 2.0, 8.0, -2.0]))
    test "in place scalar vector division":
      var v = vector([2.0, 6.0, 4.0, 16.0, -4.0])
      v /= 2.0
      check v == vector([1.0, 3.0, 2.0, 8.0, -2.0])
    test "vector sum":
      let
        v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
        w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
      check((v + w) == vector([3.0, 2.0, 4.0, 8.0, 2.0]))
    test "in place vector sum":
      var v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
      let w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
      v += w
      check v == vector([3.0, 2.0, 4.0, 8.0, 2.0])
    test "vector difference":
      let
        v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
        w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
      check((v - w) == vector([-1.0, 4.0, 0.0, 8.0, -6.0]))
    test "in place vector difference":
      var v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
      let w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
      v -= w
      check v == vector([-1.0, 4.0, 0.0, 8.0, -6.0])
    test "dot product":
      let
        v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
        w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
      check(v * w == -5.0)
    test "ℓ² norm":
      let v = vector([1.0, 1.0, 2.0, 3.0, -7.0])
      check l_2(v) == 8.0
    test "ℓ¹ norm":
      let v = vector([1.0, 1.0, 2.0, 3.0, -7.0])
      check l_1(v) == 14.0
    test "max and min of vectors":
      let v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
      check max(v) == 8.0
      check maxIndex(v) == (3, 8.0)
      check min(v) == -2.0
      check minIndex(v) == (4, -2.0)
    test "vector Hadamard multiplication":
      let
        v = vector([1.0, 3.0, 2.0, 8.0, -2.0])
        w = vector([2.0, -1.0, 2.0, 0.0, 4.0])
      check((v |*| w) == vector([2.0, -3.0, 4.0, 0.0, -8.0]))

  suite "32-bit vector operations":
    test "scalar vector multiplication":
      let v = vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
      check((v * 2'f32) == vector([2'f32, 6'f32, 4'f32, 16'f32, -4'f32]))
      check((-1'f32 * v) == vector([-1'f32, -3'f32, -2'f32, -8'f32, 2'f32]))
    test "in place scalar vector multiplication":
      var v = vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
      v *= 2'f32
      check v == vector([2'f32, 6'f32, 4'f32, 16'f32, -4'f32])
    test "scalar vector division":
      let v = vector([2'f32, 6'f32, 4'f32, 16'f32, -4'f32])
      check((v / 2'f32) == vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32]))
    test "in place scalar vector division":
      var v = vector([2'f32, 6'f32, 4'f32, 16'f32, -4'f32])
      v /= 2'f32
      check v == vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
    test "vector sum":
      let
        v = vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
        w = vector([2'f32, -1'f32, 2'f32, 0'f32, 4'f32])
      check((v + w) == vector([3'f32, 2'f32, 4'f32, 8'f32, 2'f32]))
    test "in place vector sum":
      var v = vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
      let w = vector([2'f32, -1'f32, 2'f32, 0'f32, 4'f32])
      v += w
      check v == vector([3'f32, 2'f32, 4'f32, 8'f32, 2'f32])
    test "vector difference":
      let
        v = vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
        w = vector([2'f32, -1'f32, 2'f32, 0'f32, 4'f32])
      check((v - w) == vector([-1'f32, 4'f32, 0'f32, 8'f32, -6'f32]))
    test "in place vector difference":
      var v = vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
      let w = vector([2'f32, -1'f32, 2'f32, 0'f32, 4'f32])
      v -= w
      check v == vector([-1'f32, 4'f32, 0'f32, 8'f32, -6'f32])
    test "dot product":
      let
        v = vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
        w = vector([2'f32, -1'f32, 2'f32, 0'f32, 4'f32])
      check(v * w == -5'f32)
    test "ℓ² norm":
      let v = vector([1'f32, 1'f32, 2'f32, 3'f32, -7'f32])
      check l_2(v) == 8'f32
    test "ℓ¹ norm":
      let v = vector([1'f32, 1'f32, 2'f32, 3'f32, -7'f32])
      check l_1(v) == 14'f32
    test "max and min of vectors":
      let v = vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
      check max(v) == 8'f32
      check maxIndex(v) == (3, 8'f32)
      check min(v) == -2'f32
      check minIndex(v) == (4, -2'f32)
    test "vector Hadamard multiplication":
      let
        v = vector([1'f32, 3'f32, 2'f32, 8'f32, -2'f32])
        w = vector([2'f32, -1'f32, 2'f32, 0'f32, 4'f32])
      check((v |*| w) == vector([2'f32, -3'f32, 4'f32, 0'f32, -8'f32]))

  suite "matrix/vector operations":
    test "multiplication of matrix and vector":
      let
        m = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ])
        v = vector([1.0, 3.0, 2.0, -2.0])
      check((m * v) == vector([7.0, 6.0, 5.0]))

  suite "32-bit matrix/vector operations":
    test "multiplication of matrix and vector":
      let
        m = matrix(@[
          @[1'f32, 0'f32, 2'f32, -1'f32],
          @[-1'f32, 1'f32, 3'f32, 1'f32],
          @[3'f32, 2'f32, 2'f32, 4'f32]
        ])
        v = vector([1'f32, 3'f32, 2'f32, -2'f32])
      check((m * v) == vector([7'f32, 6'f32, 5'f32]))

  suite "matrix operations":
    test "scalar matrix multiplication":
      let
        m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ])
        m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ])
      check(m1 * 3.0 == m2)
      check(3.0 * m1 == m2)
    test "in place scalar multiplication":
      var m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ])
      let m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ])
      m1 *= 3.0
      check(m1 == m2)
    test "scalar matrix division":
      let
        m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ])
        m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ])
      check(m2 / 3.0 == m1)
    test "in place scalar division":
      let m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ])
      var m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ])
      m2 /= 3.0
      check(m1 == m2)
    test "matrix sum":
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
        ])
        m3 = matrix(@[
          @[4.0, 1.0, 1.0, 0.0],
          @[1.0, 2.0, 0.0, 1.0],
          @[7.0, 3.0, 4.0, 6.0]
        ])
      check(m1 + m2 == m3)
    test "in place matrix sum":
      var m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ])
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
    test "matrix difference":
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
        ])
        m3 = matrix(@[
          @[-2.0, -1.0, 3.0, -2.0],
          @[-3.0, 0.0, 6.0, 1.0],
          @[-1.0, 1.0, 0.0, 2.0]
        ])
      check(m1 - m2 == m3)
    test "in place matrix difference":
      var m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ])
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
    test "matrix ℓ² norm":
      let m = matrix(@[
        @[1.0, 1.0, 2.0],
        @[3.0, 0.0, -7.0]
      ])
      check l_2(m) == 8.0
    test "matrix ℓ¹ norm":
      let m = matrix(@[
        @[1.0, 1.0, 2.0],
        @[3.0, 0.0, -7.0],
        @[2.5, 3.1, -1.4]
      ])
      check l_1(m) == 21.0
    test "max and min of matrices":
      let m = matrix(@[
        @[1.0, 1.0, 2.0],
        @[3.0, 0.0, -7.0]
      ])
      check max(m) == 3.0
      check min(m) == -7.0
    test "matrix multiplication":
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
        ])
        m3 = matrix(@[
          @[-10.0, -6.0, -8.0],
          @[18.0, 14.0, -2.0]
        ])
      check(m1 * m2 == m3)
    test "matrix Hadamard multiplication":
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
        ])
        m3 = matrix(@[
          @[3.0, 0.0, -2.0, -1.0],
          @[-2.0, 1.0, -9.0, 0.0],
          @[12.0, 2.0, 4.0, 8.0]
        ])
      check((m1 |*| m2) == m3)

  suite "32-bit matrix operations":
    test "scalar matrix multiplication":
      let
        m1 = matrix(@[
          @[1'f32, 3'f32],
          @[2'f32, 8'f32],
          @[-2'f32, 3'f32]
        ])
        m2 = matrix(@[
          @[3'f32, 9'f32],
          @[6'f32, 24'f32],
          @[-6'f32, 9'f32]
        ])
      check(m1 * 3 == m2)
      check(3'f32 * m1 == m2)
    test "in place scalar multiplication":
      var m1 = matrix(@[
          @[1'f32, 3'f32],
          @[2'f32, 8'f32],
          @[-2'f32, 3'f32]
        ])
      let m2 = matrix(@[
          @[3'f32, 9'f32],
          @[6'f32, 24'f32],
          @[-6'f32, 9'f32]
        ])
      m1 *= 3
      check(m1 == m2)
    test "scalar matrix division":
      let
        m1 = matrix(@[
          @[1'f32, 3'f32],
          @[2'f32, 8'f32],
          @[-2'f32, 3'f32]
        ])
        m2 = matrix(@[
          @[3'f32, 9'f32],
          @[6'f32, 24'f32],
          @[-6'f32, 9'f32]
        ])
      check(m2 / 3 == m1)
    test "in place scalar division":
      let m1 = matrix(@[
          @[1'f32, 3'f32],
          @[2'f32, 8'f32],
          @[-2'f32, 3'f32]
        ])
      var m2 = matrix(@[
          @[3'f32, 9'f32],
          @[6'f32, 24'f32],
          @[-6'f32, 9'f32]
        ])
      m2 /= 3
      check(m1 == m2)
    test "matrix sum":
      let
        m1 = matrix(@[
          @[1'f32, 0'f32, 2'f32, -1'f32],
          @[-1'f32, 1'f32, 3'f32, 1'f32],
          @[3'f32, 2'f32, 2'f32, 4'f32]
        ])
        m2 = matrix(@[
          @[3'f32, 1'f32, -1'f32, 1'f32],
          @[2'f32, 1'f32, -3'f32, 0'f32],
          @[4'f32, 1'f32, 2'f32, 2'f32]
        ])
        m3 = matrix(@[
          @[4'f32, 1'f32, 1'f32, 0'f32],
          @[1'f32, 2'f32, 0'f32, 1'f32],
          @[7'f32, 3'f32, 4'f32, 6'f32]
        ])
      check(m1 + m2 == m3)
    test "in place matrix sum":
      var m1 = matrix(@[
          @[1'f32, 0'f32, 2'f32, -1'f32],
          @[-1'f32, 1'f32, 3'f32, 1'f32],
          @[3'f32, 2'f32, 2'f32, 4'f32]
        ])
      let
        m2 = matrix(@[
          @[3'f32, 1'f32, -1'f32, 1'f32],
          @[2'f32, 1'f32, -3'f32, 0'f32],
          @[4'f32, 1'f32, 2'f32, 2'f32]
        ])
        m3 = matrix(@[
          @[4'f32, 1'f32, 1'f32, 0'f32],
          @[1'f32, 2'f32, 0'f32, 1'f32],
          @[7'f32, 3'f32, 4'f32, 6'f32]
        ])
      m1 += m2
      check m1 == m3
    test "matrix difference":
      let
        m1 = matrix(@[
          @[1'f32, 0'f32, 2'f32, -1'f32],
          @[-1'f32, 1'f32, 3'f32, 1'f32],
          @[3'f32, 2'f32, 2'f32, 4'f32]
        ])
        m2 = matrix(@[
          @[3'f32, 1'f32, -1'f32, 1'f32],
          @[2'f32, 1'f32, -3'f32, 0'f32],
          @[4'f32, 1'f32, 2'f32, 2'f32]
        ])
        m3 = matrix(@[
          @[-2'f32, -1'f32, 3'f32, -2'f32],
          @[-3'f32, 0'f32, 6'f32, 1'f32],
          @[-1'f32, 1'f32, 0'f32, 2'f32]
        ])
      check(m1 - m2 == m3)
    test "in place matrix difference":
      var m1 = matrix(@[
          @[1'f32, 0'f32, 2'f32, -1'f32],
          @[-1'f32, 1'f32, 3'f32, 1'f32],
          @[3'f32, 2'f32, 2'f32, 4'f32]
        ])
      let
        m2 = matrix(@[
          @[3'f32, 1'f32, -1'f32, 1'f32],
          @[2'f32, 1'f32, -3'f32, 0'f32],
          @[4'f32, 1'f32, 2'f32, 2'f32]
        ])
        m3 = matrix(@[
          @[-2'f32, -1'f32, 3'f32, -2'f32],
          @[-3'f32, 0'f32, 6'f32, 1'f32],
          @[-1'f32, 1'f32, 0'f32, 2'f32]
        ])
      m1 -= m2
      check m1 == m3
    test "matrix ℓ² norm":
      let m = matrix(@[
        @[1'f32, 1'f32, 2'f32],
        @[3'f32, 0'f32, -7'f32]
      ])
      check l_2(m) == 8'f32
    test "matrix ℓ¹ norm":
      let m = matrix(@[
        @[1'f32, 1'f32, 2'f32],
        @[3'f32, 0'f32, -7'f32],
        @[2.5'f32, 3.1'f32, -1.4'f32]
      ])
      check l_1(m) == 21'f32
    test "max and min of matrices":
      let m = matrix(@[
        @[1'f32, 1'f32, 2'f32],
        @[3'f32, 0'f32, -7'f32]
      ])
      check max(m) == 3'f32
      check min(m) == -7'f32
    test "matrix multiplication":
      let
        m1 = matrix(@[
          @[1'f32, 1'f32, 2'f32, -3'f32],
          @[3'f32, 0'f32, -7'f32, 2'f32]
        ])
        m2 = matrix(@[
          @[1'f32, 1'f32, 2'f32],
          @[3'f32, 1'f32, -5'f32],
          @[-1'f32, -1'f32, 2'f32],
          @[4'f32, 2'f32, 3'f32]
        ])
        m3 = matrix(@[
          @[-10'f32, -6'f32, -8'f32],
          @[18'f32, 14'f32, -2'f32]
        ])
      check(m1 * m2 == m3)
    test "matrix Hadamard multiplication":
      let
        m1 = matrix(@[
          @[1'f32, 0'f32, 2'f32, -1'f32],
          @[-1'f32, 1'f32, 3'f32, 1'f32],
          @[3'f32, 2'f32, 2'f32, 4'f32]
        ])
        m2 = matrix(@[
          @[3'f32, 1'f32, -1'f32, 1'f32],
          @[2'f32, 1'f32, -3'f32, 0'f32],
          @[4'f32, 1'f32, 2'f32, 2'f32]
        ])
        m3 = matrix(@[
          @[3'f32, 0'f32, -2'f32, -1'f32],
          @[-2'f32, 1'f32, -9'f32, 0'f32],
          @[12'f32, 2'f32, 4'f32, 8'f32]
        ])
      check((m1 |*| m2) == m3)

run()