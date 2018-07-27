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

import ./core, ./dense

type
  StaticVector*[N: static[int]; A] = distinct Vector[A]
  StaticMatrix*[M, N: static[int]; A] = distinct Matrix[A]

proc asStatic*[A](v: Vector[A], N: static[int]): auto {.inline.} =
  checkDim(v.len == N, "Wrong dimension: " & $N)
  StaticVector[N, A](v)

proc asStatic*[A](m: Matrix[A], M, N: static[int]): auto {.inline.} =
  checkDim(m.M == M, "Wrong dimension: " & $M)
  checkDim(m.N == N, "Wrong dimension: " & $N)
  StaticMatrix[M, N, A](m)

proc asDynamic*[N: static[int]; A](v: StaticVector[N, A]): auto {.inline.} =
  Vector[A](v)

proc asDynamic*[M, N: static[int]; A](m: StaticMatrix[M, N, A]): auto {.inline.} =
  Matrix[A](m)

template dyn(v: StaticVector, A: typedesc): Vector[A] =
  Vector[A](v)

template dyn(m: StaticMatrix, A: typedesc): Matrix[A] =
  Matrix[A](m)

proc `==`*[N: static[int]; A](v, w: StaticVector[N, A]): bool {.inline.} =
  dyn(v, A) == dyn(w, A)

proc `==`*[M, N: static[int]; A](m, n: StaticMatrix[M, N, A]): bool {.inline.} =
  dyn(m, A) == dyn(n, A)

proc `$`*[N: static[int]; A](v: StaticVector[N, A]): string =
  $(Vector[A](v))

proc `$`*[M, N: static[int]; A](m: StaticMatrix[M, N, A]): string =
  $(Matrix[A](m))

type Array[N: static[int]; A] = array[N, A]
type DoubleArray[M, N: static[int], A] = array[M, array[N, A]]

proc makeVector*[A](N: static[int], f: proc (i: int): A): auto =
  dense.makeVector(N, f).asStatic(N)

template makeVectorI*[A](N: static[int], f: untyped): auto =
  dense.makeVectorI[A](N, f).asStatic(N)

proc vector*[N: static[int]; A](v: Array[N, A]): auto =
  dense.vector(v).asStatic(N)

proc constantVector*[A](N: static[int], x: A): auto =
  dense.constantVector(N, x).asStatic(N)

proc randomVector*(N: static[int], max: float64 = 1): auto =
  dense.randomVector(N, max).asStatic(N)

proc randomVector*(N: static[int], max: float32): auto =
  dense.randomVector(N, max).asStatic(N)

proc zeros*(N: static[int]): auto = constantVector(N, 0'f64)

proc zeros*(N: static[int], A: typedesc[float32]): auto = constantVector(N, 0'f32)

proc zeros*(N: static[int], A: typedesc[float64]): auto = constantVector(N, 0'f64)

proc ones*(N: static[int]): auto = constantVector(N, 1'f64)

proc ones*(N: static[int], A: typedesc[float32]): auto = constantVector(N, 1'f32)

proc ones*(N: static[int], A: typedesc[float64]): auto = constantVector(N, 1'f64)

proc constantMatrix*[A](M, N: static[int], x: A): auto =
  dense.constantMatrix(M, N, x).asStatic(M, N)

proc makeMatrix*[A](M, N: static[int], f: proc (i, j: int): A): auto =
  dense.makeMatrix(M, N, f).asStatic(M, N)

template makeMatrixIJ*(A: typedesc; M, N: static[int], f: untyped): auto =
  dense.makeMatrixIJ(A, M, N, f).asStatic(M, N)

proc zeros*(M, N: static[int]): auto = constantMatrix(M, N, 0'f64)

proc zeros*(M, N: static[int], A: typedesc[float32]): auto = constantMatrix(M, N, 0'f32)

proc zeros*(M, N: static[int], A: typedesc[float64]): auto = constantMatrix(M, N, 0'f64)

proc ones*(M, N: static[int]): auto = constantMatrix(M, N, 1'f64)

proc ones*(M, N: static[int], A: typedesc[float32]): auto = constantMatrix(M, N, 1'f32)

proc ones*(M, N: static[int], A: typedesc[float64]): auto = constantMatrix(M, N, 1'f64)

proc eye*(N: static[int]): auto =
  dense.eye(N).asStatic(N, N)

proc eye*(N: static[int], A: typedesc[float32]): auto =
  dense.eye(N, A).asStatic(N, N)

proc eye*(N: static[int], A: typedesc[float64]): auto =
  dense.eye(N, A).asStatic(N, N)

# Accessors

proc len*[N: static[int]; A](v: StaticVector[N, A]): int {. inline .} = N

proc `[]`*[N: static[int]; A](v: StaticVector[N, A], i: int): A {. inline .} =
  dyn(v, A)[i]

proc `[]=`*[N: static[int]; A](v: StaticVector[N, A], i: int, val: A) {. inline .} =
  dyn(v, A)[i] = val

proc `[]`*[M, N: static[int]; A](m: StaticMatrix[M, N, A], i, j: int): A {. inline .} =
  dyn(m, A)[i, j]

proc `[]=`*[M, N: static[int]; A](v: StaticMatrix[M, N, A], i, j: int, val: A) {. inline .} =
  dyn(m, A)[i, j] = val

proc dim*[M, N: static[int]; A](m: StaticMatrix[M, N, A]): tuple[M, N: int] {. inline .} =
  (M: M, N: N)

# Operations

proc `*`*[M, N, K: static[int]; A: SomeFloat](
  m: StaticMatrix[M, K, A],
  n: StaticMatrix[K, N, A]
): StaticMatrix[M, N, A] = (dyn(m, A) * dyn(n, A)).asStatic(M, N)

proc randomStaticMatrix*(M, N: static[int]): auto =
  randomMatrix(M, N).asStatic(M, N)