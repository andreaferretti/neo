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

import times, neo

proc matrixMult() =
  let
    m1 = randomMatrix(1000, 987)
    m2 = randomMatrix(987, 876)
    startTime = epochTime()

  for i in 0 ..< 100:
    discard m1 * m2
  let endTime = epochTime()

  echo "We have required ", endTime - startTime, " seconds to multiply matrices 100 times."

proc columnIteration() =
  let m = randomMatrix(1000, 987)
  var v = randomVector(1000)
  let startTime = epochTime()

  for col in m.columns:
    v += col
  let endTime = epochTime()

  echo "We have required ", endTime - startTime, " seconds to iterate 987 columns."

proc columnIterationSlow() =
  let m = randomMatrix(1000, 987)
  var v = randomVector(1000)
  let startTime = epochTime()

  for col in m.columnsSlow:
    v += col
  let endTime = epochTime()

  echo "We have required ", endTime - startTime, " seconds to iterate 987 columns slowly."

proc main() =
  matrixMult()
  columnIteration()
  columnIterationSlow()

when isMainModule:
  main()