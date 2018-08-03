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
  suite "collection operations":
    test "cumulative sum over 32-bit vectors":
      let
        v = statics.vector([1'f32, 3.5'f32, 2'f32, 4.5'f32])
        w = cumsum(v)
      check w == statics.vector([1'f32, 4.5'f32, 6.5'f32, 11'f32])
    test "cumulative sum over 64-bit vectors":
      let
        v = statics.vector([1.0, 3.5, 2.0, 4.5])
        w = cumsum(v)
      check w == statics.vector([1.0, 4.5, 6.5, 11.0])
    test "sum over 32-bit vectors":
      let v = statics.vector([1'f32, 3.5'f32, 2'f32, 4.5'f32])
      check v.sum == 11'f32
    test "sum over 64-bit vectors":
      let v = statics.vector([1.0, 3.5, 2.0, 4.5])
      check v.sum == 11.0
    test "mean over 32-bit vectors":
      let v = statics.vector([1'f32, 3.5'f32, 2'f32, 4.5'f32])
      check v.mean == 2.75'f32
    test "mean over 64-bit vectors":
      let v = statics.vector([1.0, 3.5, 2.0, 4.5])
      check v.mean == 2.75
    test "variance over 32-bit vectors":
      let v = statics.vector([2'f32, 4'f32, 4'f32, 4'f32, 5'f32, 5'f32, 7'f32, 9'f32])
      check v.variance == 4'f32
    test "variance over 64-bit vectors":
      let v = statics.vector([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
      check v.variance == 4.0
    test "standard deviation over 32-bit vectors":
      let v = statics.vector([2'f32, 4'f32, 4'f32, 4'f32, 5'f32, 5'f32, 7'f32, 9'f32])
      check v.stddev == 2'f32
    test "standard deviation over 64-bit vectors":
      let v = statics.vector([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
      check v.stddev == 2.0
    test "sum over 32-bit matrices":
      let m = statics.matrix([[1'f32, 3.5'f32], [2'f32, 4.5'f32]])
      check m.sum == 11'f32
    test "sum over 64-bit matrices":
      let m = statics.matrix([[1.0, 3.5], [2.0, 4.5]])
      check m.sum == 11.0
    test "mean over 32-bit matrices":
      let m = statics.matrix([[1'f32, 3.5'f32], [2'f32, 4.5'f32]])
      check m.mean == 2.75'f32
    test "mean over 64-bit matrices":
      let m = statics.matrix([[1.0, 3.5], [2.0, 4.5]])
      check m.mean == 2.75

run()