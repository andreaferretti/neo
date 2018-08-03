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

import unittest, sequtils, neo

proc run() =
  suite "conversions to dense":
    let expected = matrix(@[
      @[1.0, 0.0, 2.0, 3.0],
      @[0.0, 4.0, 0.0, 0.0],
      @[5.0, 0.0, 6.0, 7.0],
      @[0.0, 8.0, 0.0, 9.0]
    ])

    test "sparse vector conversion":
      let v = sparseVector(10, @[3'i32, 5, 7], @[2.0, 3, -1])
      check v.dense == vector([0.0, 0, 0, 2, 0, 3, 0, -1, 0, 0])
    test "csr matrix conversion":
      let m = csr(
        rows = @[0'i32, 3, 4, 7, 9],
        cols = @[0'i32, 2, 3, 1, 0, 2, 3, 1, 3],
        vals = @[1'f64, 2, 3, 4, 5, 6, 7, 8, 9],
        numCols = 4
      )
      check m.dense == expected
    test "csc matrix conversion":
      let m = csc(
        rows = @[0'i32, 2, 1, 3, 0, 2, 0, 2, 3],
        cols = @[0'i32, 2, 4, 6, 9],
        vals = @[1'f64, 5, 4, 8, 2, 6, 3, 7, 9],
        numRows = 4
      )
      check m.dense == expected
    test "coo matrix conversion":
      let m = coo(
        rows = @[0'i32, 0, 0, 1, 2, 2, 2, 3, 3],
        cols = @[0'i32, 2, 3, 1, 0, 2, 3, 1, 3],
        vals = @[1'f64, 2, 3, 4, 5, 6, 7, 8, 9],
        numRows = 4,
        numCols = 4
      )
      check m.dense == expected

run()