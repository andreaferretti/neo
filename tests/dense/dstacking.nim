# Copyright 2018 UniCredit S.p.A.
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
  suite "stacking of vectors":
    test "concat (horizontal stack) of two vectors":
      let
        v1 = vector([1.0, 2.0])
        v2 = vector([5.0, 7.0, 9.0])
        v = vector([1.0, 2.0, 5.0, 7.0, 9.0])
      check concat(v1, v2) == v
      check hstack(v1, v2) == v

    test "concat (horizontal stack) of three vectors":
      let
        v1 = vector([1.0, 2.0])
        v2 = vector([5.0, 7.0, 9.0])
        v3 = vector([9.9, 8.8, 7.7, 6.6])
        v = vector([1.0, 2.0, 5.0, 7.0, 9.0, 9.9, 8.8, 7.7, 6.6])
      check concat(v1, v2, v3) == v
      check hstack(v1, v2, v3) == v

    test "concat (horizontal stack) of two non-contiguous vectors":
      let
        v1 = matrix(@[
          @[1.0, 0.0],
          @[2.0, 0.0]
        ], order=rowMajor).column(0)
        v2 = matrix(@[
          @[5.0, 0.0],
          @[7.0, 0.0],
          @[9.0, 0.0]
        ], order=rowMajor).column(0)
        v = vector([1.0, 2.0, 5.0, 7.0, 9.0])
      check concat(v1, v2) == v
      check hstack(v1, v2) == v

    test "horizontal stack of two matrices":
      let
        m1 = matrix(@[
          @[1.0, 2.0],
          @[3.0, 4.0]
        ])
        m2 = matrix(@[
          @[5.0, 7.0, 9.0],
          @[6.0, 2.0, 1.0]
        ])
        m = matrix(@[
          @[1.0, 2.0, 5.0, 7.0, 9.0],
          @[3.0, 4.0, 6.0, 2.0, 1.0]
        ])
      check hstack(m1, m2) == m

    test "horizontal stack of three matrices":
      let
        m1 = matrix(@[
          @[1.0, 2.0],
          @[3.0, 4.0]
        ])
        m2 = matrix(@[
          @[5.0, 7.0, 9.0],
          @[6.0, 2.0, 1.0]
        ])
        m3 = matrix(@[
          @[2.0, 2.0],
          @[1.0, 3.0]
        ])
        m = matrix(@[
          @[1.0, 2.0, 5.0, 7.0, 9.0, 2.0, 2.0],
          @[3.0, 4.0, 6.0, 2.0, 1.0, 1.0, 3.0]
        ])
      check hstack(m1, m2, m3) == m

    test "vertical stack of two vectors":
      let
        v1 = vector([1.0, 2.0, 3.0])
        v2 = vector([5.0, 7.0, 9.0])
        m = matrix(@[
          @[1.0, 2.0, 3.0],
          @[5.0, 7.0, 9.0]
        ])
      check matrix(@[v1, v2]) == m
      check vstack(v1, v2) == m

    test "vertical stack of three vectors":
      let
        v1 = vector([1.0, 2.0, 3.0])
        v2 = vector([5.0, 7.0, 9.0])
        v3 = vector([9.9, 8.8, 7.7])
        m = matrix(@[
          @[1.0, 2.0, 3.0],
          @[5.0, 7.0, 9.0],
          @[9.9, 8.8, 7.7]
        ])
      check matrix(@[v1, v2, v3]) == m
      check vstack(v1, v2, v3) == m

    test "vertical stack of three matrices":
      let
        m1 = matrix(@[
          @[1.0, 2.0],
          @[3.0, 4.0]
        ]).T
        m2 = matrix(@[
          @[5.0, 7.0, 9.0],
          @[6.0, 2.0, 1.0]
        ]).T
        m3 = matrix(@[
          @[2.0, 2.0],
          @[1.0, 3.0]
        ]).T
        m = matrix(@[
          @[1.0, 2.0, 5.0, 7.0, 9.0, 2.0, 2.0],
          @[3.0, 4.0, 6.0, 2.0, 1.0, 1.0, 3.0]
        ]).T
      check vstack(m1, m2, m3) == m

run()
