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

type
  Complex*[A] = tuple[re, im: A]
  Scalar* = float32 or float64 or Complex[float32] or Complex[float64]
  UncheckedArray*{.unchecked.}[A] = array[1, A]
  CPointer*[A] = ptr UncheckedArray[A]

type
  DimensionError* = object of ValueError
  OutOfBoundsError* = object of ValueError
  LinearAlgebraError* = object of FloatingPointError

template checkDim*(cond: untyped, msg = "") =
  when compileOption("assertions"):
    {.line.}:
      if not cond:
        raise newException(DimensionError, msg)

template checkBounds*(cond: untyped, msg = "") =
  when compileOption("assertions"):
    {.line.}:
      if not cond:
        raise newException(OutOfBoundsError, msg)