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

import unittest, neo/dense, sequtils

proc run() =
  suite "Matrix Decompositions":
      test "SVD":
        let a = @[@[  7.52, -1.10, -7.95,  1.08],
                @[ -0.76,  0.62,  9.34, -7.10],
                @[  5.13,  6.62, -5.66,  0.87],
                @[ -4.75,  8.52,  5.75,  5.30],
                @[  1.33,  4.91, -5.49, -3.52],
                @[ -2.40, -6.77,  2.34,  3.95]].matrix
    
        let (U, S, Vh) = svd(a)
        check U * diag(S.toSeq) * Vh =~ a

run()  
