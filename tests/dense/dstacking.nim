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
        v1 = vector([1.0, 2.0, 3.0])
        v2 = vector([5.0, 7.0, 9.0])
        v = vector([1.0, 2.0, 3.0, 5.0, 7.0, 9.0])
      check concat(v1, v2) == v
      check hstack(v1, v2) == v

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


run()