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
  suite "trivial operations on 32-bit matrices":
    test "reshape of matrices":
      let
        m1 = statics.matrix([
          [1'f32, 0'f32, 2'f32, -1'f32],
          [-1'f32, 1'f32, 3'f32, 1'f32],
          [3'f32, 2'f32, 2'f32, 4'f32]
        ])
        m2 = statics.matrix([
          [1'f32, 1'f32, 2'f32],
          [-1'f32, 2'f32, -1'f32],
          [3'f32, 2'f32, 1'f32],
          [0'f32, 3'f32, 4'f32]
        ])
      check m1.reshape(4, 3) == m2
    test "turn vectors into matrices":
      let
        v = statics.vector([1'f32, -1'f32, 3'f32, 0'f32, 1'f32, 2'f32, 2'f32, 3'f32, 2'f32, -1'f32, 1'f32, 4'f32])
        m = statics.matrix([
          [1'f32, 0'f32, 2'f32, -1'f32],
          [-1'f32, 1'f32, 3'f32, 1'f32],
          [3'f32, 2'f32, 2'f32, 4'f32]
        ])
      check v.asMatrix(3, 4) == m
    test "turn matrices into vectors":
      let
        v = statics.vector([1'f32, -1'f32, 3'f32, 0'f32, 1'f32, 2'f32, 2'f32, 3'f32, 2'f32, -1'f32, 1'f32, 4'f32])
        m = statics.matrix([
          [1'f32, 0'f32, 2'f32, -1'f32],
          [-1'f32, 1'f32, 3'f32, 1'f32],
          [3'f32, 2'f32, 2'f32, 4'f32]
        ])
      check m.asVector == v
    test "transpose of matrices":
      let
        m1 = statics.matrix([
          [1'f32, 0'f32, 2'f32, -1'f32],
          [-1'f32, 1'f32, 3'f32, 1'f32],
          [3'f32, 2'f32, 2'f32, 4'f32]
        ])
        m2 = statics.matrix([
          [1'f32, -1'f32, 3'f32],
          [0'f32, 1'f32, 2'f32],
          [2'f32, 3'f32, 2'f32],
          [-1'f32, 1'f32, 4'f32]
        ])
      check m1.t == m2
    test "hard transpose of matrices":
      let m = statics.matrix([
        [1'f32, 0'f32, 2'f32, -1'f32],
        [-1'f32, 1'f32, 3'f32, 1'f32],
        [3'f32, 2'f32, 2'f32, 4'f32]
      ])

      check(m.t == m.T)
    test "hard transpose of row major matrices":
      let m = statics.matrix([
        [1'f32, 0'f32, 2'f32, -1'f32],
        [-1'f32, 1'f32, 3'f32, 1'f32],
        [3'f32, 2'f32, 2'f32, 4'f32]
      ], order = rowMajor)

      check(m.t == m.T)

  suite "trivial operations should share storage":
    test "reshape of matrices":
      var
        m1 = statics.matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ])
        m2 = m1.reshape(4, 3)
      m2[2, 1] = 0.0
      check m1[0, 2] == 0.0
    test "turn vectors into matrices":
      var
        v = statics.vector([1.0, -1.0, 3.0, 0.0, 1.0, 2.0, 2.0, 3.0, 2.0, -1.0, 1.0, 4.0])
        m = v.asMatrix(3, 4)
      m[2, 1] = 0.0
      check v[5] == 0.0
    test "turn matrices into vectors":
      var
        m = statics.matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ])
        v = m.asVector
      v[5] = 0.0
      check m[2, 1] == 0.0
    test "transpose of matrices":
      var
        m1 = statics.matrix([
          [1.0, 0.0, 2.0, -1.0],
          [-1.0, 1.0, 3.0, 1.0],
          [3.0, 2.0, 2.0, 4.0]
        ])
        m2 = m1.t
      m2[1, 2] = 0.0
      check m1[2, 1] == 0.0

run()