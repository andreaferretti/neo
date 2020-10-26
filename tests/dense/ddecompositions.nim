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

proc `~=`(a,b:SomeFloat): bool =
  abs(a-b) < 1e-2


proc run() =
  suite "Matrix Decompositions":
      test "SVD":
        let a = @[@[  7.52, -1.10, -7.95,  1.08],
                @[ -0.76,  0.62,  9.34, -7.10],
                @[  5.13,  6.62, -5.66,  0.87],
                @[ -4.75,  8.52,  5.75,  5.30],
                @[  1.33,  4.91, -5.49, -3.52],
                @[ -2.40, -6.77,  2.34,  3.95]].matrix

        let expected_U =  @[@[-0.57,  0.18,  0.01,  0.53],
                          @[ 0.46, -0.11, -0.72,  0.42],
                          @[-0.45, -0.41,  0.00,  0.36],
                          @[ 0.33, -0.69,  0.49,  0.19],
                          @[-0.32, -0.31, -0.28, -0.61],
                          @[ 0.21,  0.46,  0.39,  0.09]].matrix

        let expected_S = @[@[18.37, 13.63, 10.85, 4.49]].matrix

        let expected_Vh = @[@[-0.52, -0.12,  0.85, -0.03],
                          @[ 0.08, -0.99, -0.09, -0.01],
                          @[-0.28, -0.02, -0.14,  0.95],
                          @[ 0.81,  0.01,  0.50,  0.31]].matrix
    
        let (U, S, Vh) = svd(a)
        for pos,val in U.pairs:
          check val ~= expected_U[pos[0],pos[1]]
        for pos,val in S.pairs:
          check val ~= expected_S[pos[0],pos[1]]
        for pos,val in Vh.pairs:
          check val ~= expected_Vh[pos[0],pos[1]]

run()  