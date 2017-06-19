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


suite "rewrite macros tests":
  test "vector linear combination":
    resetRewriteCount()
    let
      a = vector(1.0, 2.0, 3.0)
      b = vector(2.0, -1.0, 3.0)

    discard a + 3.0 * b

    check getRewriteCount() == 1

  test "mutable vector linear combination":
    resetRewriteCount()
    var a = vector(1.0, 2.0, 3.0)
    let b = vector(2.0, -1.0, 3.0)

    a += 3.0 * b

    check getRewriteCount() == 1

  test "vector linear combination after template application":
    resetRewriteCount()
    let
      a = vector(1.0, 2.0, 3.0)
      b = vector(2.0, -1.0, 3.0)

    discard a + b / 3.0

    check getRewriteCount() == 1