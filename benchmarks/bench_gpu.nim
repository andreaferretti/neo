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

import times, neo, neo/cuda

proc main() =
  echo "We are about to perform 100 times matrix/vector multiplication with size 10000x10000"
  let
    m = randomMatrix(10000, 10000, max = 1'f32)
    v = randomVector(10000, max = 1'f32)
    m1 = m.gpu()
    v1 = v.gpu()

  let startTime = epochTime()
  for i in 0 .. < 100:
    discard m * v
  let endTime = epochTime()

  echo "We have required ", endTime - startTime, " seconds on the CPU."

  let startTime1 = epochTime()
  for i in 0 .. < 100:
    discard m1 * v1
  let endTime1 = epochTime()

  echo "We have required ", endTime1 - startTime1, " seconds on the GPU."

when isMainModule:
  main()