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
  suite "32-bit vector operations":
    test "scalar vector multiplication":
      let
        v1 = randomVector(10, max=1'f32)
        p1 = v1.gpu()
      check(v1 * 3 == (p1 * 3).cpu())
      check(2'f32 * v1 == (2'f32 * p1).cpu())
    test "in place scalar vector multiplication":
      var
        v1 = randomVector(10, max=1'f32)
        p1 = v1.gpu()
      v1 *= 5
      p1 *= 5
      check(v1 == p1.cpu())
    test "vector sum":
      let
        v1 = randomVector(10, max=1'f32)
        v2 = randomVector(10, max=1'f32)
        p1 = v1.gpu()
        p2 = v2.gpu()
        p3 = p1 + p2
        v3 = p3.cpu()
      check(v1 + v2 == v3)
    test "in place vector sum":
      var v1 = randomVector(10, max=1'f32)
      let v2 = randomVector(10, max=1'f32)
      var p1 = v1.gpu()
      let p2 = v2.gpu()
      v1 += v2
      p1 += p2
      check(v1 == p1.cpu())
    test "vector difference":
      let
        v1 = randomVector(10, max=1'f32)
        v2 = randomVector(10, max=1'f32)
        p1 = v1.gpu()
        p2 = v2.gpu()
        p3 = p1 - p2
        v3 = p3.cpu()
      check(v1 - v2 == v3)
    test "in place vector difference":
      var v1 = randomVector(10, max=1'f32)
      let v2 = randomVector(10, max=1'f32)
      var p1 = v1.gpu()
      let p2 = v2.gpu()
      v1 -= v2
      p1 -= p2
      check(v1 == p1.cpu())
    test "dot product":
      let
        v = vector(1.0, 3.0, 2.0, 8.0, -2.0).to32().gpu()
        w = vector(2.0, -1.0, 2.0, 0.0, 4.0).to32().gpu()
      check(v * w == -5.0)
    test "ℓ² norm":
      let v = vector([1.0, 1.0, 2.0, 3.0, -7.0]).to32().gpu()
      check l_2(v) == 8.0
    test "ℓ¹ norm":
      let v = vector([1.0, 1.0, 2.0, 3.0, -7.0]).to32().gpu()
      check l_1(v) == 14.0

  suite "64-bit vector operations":
    test "scalar vector multiplication":
      let
        v1 = randomVector(10, max=1.0)
        p1 = v1.gpu()
        p2 = 2.0 * p1
        p3 = p1 * 3
      check(v1 * 3 == p3.cpu())
      check(2.0 * v1 == p2.cpu())
    test "in place scalar vector multiplication":
      var
        v1 = randomVector(10, max=1.0)
        p1 = v1.gpu()
      v1 *= 5
      p1 *= 5
      check(v1 == p1.cpu())
    test "vector sum":
      let
        v1 = randomVector(10, max=1.0)
        v2 = randomVector(10, max=1.0)
        p1 = v1.gpu()
        p2 = v2.gpu()
        p3 = p1 + p2
        v3 = p3.cpu()
      check(v1 + v2 == v3)
    test "in place vector sum":
      var v1 = randomVector(10, max=1.0)
      let v2 = randomVector(10, max=1.0)
      var p1 = v1.gpu()
      let p2 = v2.gpu()
      v1 += v2
      p1 += p2
      check(v1 == p1.cpu())
    test "vector difference":
      let
        v1 = randomVector(10, max=1.0)
        v2 = randomVector(10, max=1.0)
        p1 = v1.gpu()
        p2 = v2.gpu()
        p3 = p1 - p2
        v3 = p3.cpu()
      check(v1 - v2 == v3)
    test "in place vector difference":
      var v1 = randomVector(10, max=1.0)
      let v2 = randomVector(10, max=1.0)
      var p1 = v1.gpu()
      let p2 = v2.gpu()
      v1 -= v2
      p1 -= p2
      check(v1 == p1.cpu())
    test "dot product":
      let
        v = vector([1.0, 3.0, 2.0, 8.0, -2.0]).gpu()
        w = vector([2.0, -1.0, 2.0, 0.0, 4.0]).gpu()
      check(v * w == -5.0)
    test "ℓ² norm":
      let v = vector([1.0, 1.0, 2.0, 3.0, -7.0]).gpu()
      check l_2(v) == 8.0
    test "ℓ¹ norm":
      let v = vector([1.0, 1.0, 2.0, 3.0, -7.0]).gpu()
      check l_1(v) == 14.0

  suite "matrix/vector operations":
    test "multiplication of 32-bit matrix and vector":
      let
        m = matrix(@[
          @[1'f32, 0'f32, 2'f32, -1'f32],
          @[-1'f32, 1'f32, 3'f32, 1'f32],
          @[3'f32, 2'f32, 2'f32, 4'f32]
        ]).gpu()
        v = vector([1'f32, 3'f32, 2'f32, -2'f32]).gpu()

      check((m * v).cpu() == vector([7'f32, 6'f32, 5'f32]))
    test "multiplication of 64-bit matrix and vector":
      let
        m = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).gpu()
        v = vector([1.0, 3.0, 2.0, -2.0]).gpu()

      check((m * v).cpu() == vector([7.0, 6.0, 5.0]))

  suite "32-bit matrix operations":
    test "scalar matrix multiplication":
      let
        m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ]).to32().gpu()
        m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ]).to32().gpu()
      check(m1 * 3'f32 == m2)
      check(3'f32 * m1 == m2)
    test "in place scalar multiplication":
      var m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ]).to32().gpu()
      let m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ]).to32().gpu()
      m1 *= 3.0
      check(m1 == m2)
    test "scalar matrix division":
      let
        m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ]).to32().gpu()
        m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ]).to32().gpu()
      check(m2 / 3.0 == m1)
    test "in place scalar division":
      let m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ]).to32().gpu()
      var m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ]).to32().gpu()
      m2 /= 3.0
      check(m1 == m2)
    test "matrix sum":
      let
        m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).to32().gpu()
        m2 = matrix(@[
          @[3.0, 1.0, -1.0, 1.0],
          @[2.0, 1.0, -3.0, 0.0],
          @[4.0, 1.0, 2.0, 2.0]
        ]).to32().gpu()
        m3 = matrix(@[
          @[4.0, 1.0, 1.0, 0.0],
          @[1.0, 2.0, 0.0, 1.0],
          @[7.0, 3.0, 4.0, 6.0]
        ]).to32().gpu()
      check(m1 + m2 == m3)
    test "in place matrix sum":
      var m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).to32().gpu()
      let
        m2 = matrix(@[
          @[3.0, 1.0, -1.0, 1.0],
          @[2.0, 1.0, -3.0, 0.0],
          @[4.0, 1.0, 2.0, 2.0]
        ]).to32().gpu()
        m3 = matrix(@[
          @[4.0, 1.0, 1.0, 0.0],
          @[1.0, 2.0, 0.0, 1.0],
          @[7.0, 3.0, 4.0, 6.0]
        ]).to32().gpu()
      m1 += m2
      check m1 == m3
    test "matrix difference":
      let
        m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).to32().gpu()
        m2 = matrix(@[
          @[3.0, 1.0, -1.0, 1.0],
          @[2.0, 1.0, -3.0, 0.0],
          @[4.0, 1.0, 2.0, 2.0]
        ]).to32().gpu()
        m3 = matrix(@[
          @[-2.0, -1.0, 3.0, -2.0],
          @[-3.0, 0.0, 6.0, 1.0],
          @[-1.0, 1.0, 0.0, 2.0]
        ]).to32().gpu()
      check(m1 - m2 == m3)
    test "in place matrix difference":
      var m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).to32().gpu()
      let
        m2 = matrix(@[
          @[3.0, 1.0, -1.0, 1.0],
          @[2.0, 1.0, -3.0, 0.0],
          @[4.0, 1.0, 2.0, 2.0]
        ]).to32().gpu()
        m3 = matrix(@[
          @[-2.0, -1.0, 3.0, -2.0],
          @[-3.0, 0.0, 6.0, 1.0],
          @[-1.0, 1.0, 0.0, 2.0]
        ]).to32().gpu()
      m1 -= m2
      check m1 == m3
    test "matrix ℓ² norm":
      let m = matrix(@[
        @[1'f32, 1'f32, 2'f32],
        @[3'f32, 0'f32, -7'f32]
      ]).gpu()
      check l_2(m) == 8'f32
    test "matrix ℓ¹ norm":
      let m = matrix(@[
        @[1'f32, 1'f32, 2'f32],
        @[3'f32, 0'f32, -7'f32],
        @[2.5'f32, 3.1'f32, -1.4'f32]
      ]).gpu()
      check l_1(m) == 21'f32
    test "matrix multiplication":
      let
        m1 = matrix(@[
          @[1'f32, 1'f32, 2'f32, -3'f32],
          @[3'f32, 0'f32, -7'f32, 2'f32]
        ]).gpu()
        m2 = matrix(@[
          @[1'f32, 1'f32, 2'f32],
          @[3'f32, 1'f32, -5'f32],
          @[-1'f32, -1'f32, 2'f32],
          @[4'f32, 2'f32, 3'f32]
        ]).gpu()
        m3 = matrix(@[
          @[-10'f32, -6'f32, -8'f32],
          @[18'f32, 14'f32, -2'f32]
        ]).gpu()
      check(m1 * m2 == m3)

  suite "64-bit matrix operations":
    test "scalar matrix multiplication":
      let
        m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ]).gpu()
        m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ]).gpu()
      check(m1 * 3.0 == m2)
      check(3.0 * m1 == m2)
    test "in place scalar multiplication":
      var m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ]).gpu()
      let m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ]).gpu()
      m1 *= 3.0
      check(m1 == m2)
    test "scalar matrix division":
      let
        m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ]).gpu()
        m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ]).gpu()
      check(m2 / 3.0 == m1)
    test "in place scalar division":
      let m1 = matrix(@[
          @[1.0, 3.0],
          @[2.0, 8.0],
          @[-2.0, 3.0]
        ]).gpu()
      var m2 = matrix(@[
          @[3.0, 9.0],
          @[6.0, 24.0],
          @[-6.0, 9.0]
        ]).gpu()
      m2 /= 3.0
      check(m1 == m2)
    test "matrix sum":
      let
        m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).gpu()
        m2 = matrix(@[
          @[3.0, 1.0, -1.0, 1.0],
          @[2.0, 1.0, -3.0, 0.0],
          @[4.0, 1.0, 2.0, 2.0]
        ]).gpu()
        m3 = matrix(@[
          @[4.0, 1.0, 1.0, 0.0],
          @[1.0, 2.0, 0.0, 1.0],
          @[7.0, 3.0, 4.0, 6.0]
        ]).gpu()
      check(m1 + m2 == m3)
    test "in place matrix sum":
      var m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).gpu()
      let
        m2 = matrix(@[
          @[3.0, 1.0, -1.0, 1.0],
          @[2.0, 1.0, -3.0, 0.0],
          @[4.0, 1.0, 2.0, 2.0]
        ]).gpu()
        m3 = matrix(@[
          @[4.0, 1.0, 1.0, 0.0],
          @[1.0, 2.0, 0.0, 1.0],
          @[7.0, 3.0, 4.0, 6.0]
        ]).gpu()
      m1 += m2
      check m1 == m3
    test "matrix difference":
      let
        m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).gpu()
        m2 = matrix(@[
          @[3.0, 1.0, -1.0, 1.0],
          @[2.0, 1.0, -3.0, 0.0],
          @[4.0, 1.0, 2.0, 2.0]
        ]).gpu()
        m3 = matrix(@[
          @[-2.0, -1.0, 3.0, -2.0],
          @[-3.0, 0.0, 6.0, 1.0],
          @[-1.0, 1.0, 0.0, 2.0]
        ]).gpu()
      check(m1 - m2 == m3)
    test "in place matrix difference":
      var m1 = matrix(@[
          @[1.0, 0.0, 2.0, -1.0],
          @[-1.0, 1.0, 3.0, 1.0],
          @[3.0, 2.0, 2.0, 4.0]
        ]).gpu()
      let
        m2 = matrix(@[
          @[3.0, 1.0, -1.0, 1.0],
          @[2.0, 1.0, -3.0, 0.0],
          @[4.0, 1.0, 2.0, 2.0]
        ]).gpu()
        m3 = matrix(@[
          @[-2.0, -1.0, 3.0, -2.0],
          @[-3.0, 0.0, 6.0, 1.0],
          @[-1.0, 1.0, 0.0, 2.0]
        ]).gpu()
      m1 -= m2
      check m1 == m3
    test "matrix ℓ² norm":
      let m = matrix(@[
        @[1.0, 1.0, 2.0],
        @[3.0, 0.0, -7.0]
      ]).gpu()
      check l_2(m) == 8.0
    test "matrix ℓ¹ norm":
      let m = matrix(@[
        @[1.0, 1.0, 2.0],
        @[3.0, 0.0, -7.0],
        @[2.5, 3.1, -1.4]
      ]).gpu()
      check l_1(m) == 21.0
    test "matrix multiplication":
      let
        m1 = matrix(@[
          @[1.0, 1.0, 2.0, -3.0],
          @[3.0, 0.0, -7.0, 2.0]
        ]).gpu()
        m2 = matrix(@[
          @[1.0, 1.0, 2.0],
          @[3.0, 1.0, -5.0],
          @[-1.0, -1.0, 2.0],
          @[4.0, 2.0, 3.0]
        ]).gpu()
        m3 = matrix(@[
          @[-10.0, -6.0, -8.0],
          @[18.0, 14.0, -2.0]
        ]).gpu()
      check(m1 * m2 == m3)

run()