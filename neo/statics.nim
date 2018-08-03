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

proc `=~`*[N: static[int]; A](v, w: StaticVector[N, A]): bool {.inline.} =
  dyn(v, A) =~ dyn(w, A)

proc `=~`*[M, N: static[int]; A](m, n: StaticMatrix[M, N, A]): bool {.inline.} =
  dyn(m, A) =~ dyn(n, A)

template `!=~`*(a, b: StaticVector or StaticMatrix): bool =
  not(a =~ b)

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

proc constantMatrix*[A](M, N: static[int], x: A, order = colMajor): auto =
  dense.constantMatrix(M, N, x, order).asStatic(M, N)

proc makeMatrix*[A](M, N: static[int], f: proc (i, j: int): A, order = colMajor): auto =
  dense.makeMatrix(M, N, f, order).asStatic(M, N)

template makeMatrixIJ*(A: typedesc; M, N: static[int], f: untyped, order = colMajor): auto =
  dense.makeMatrixIJ(A, M, N, f, order).asStatic(M, N)

proc matrix*[M, N: static[int], A](xs: DoubleArray[M, N, A], order = colMajor): auto =
  makeMatrixIJ(A, M, N, xs[i][j], order)

proc zeros*(M, N: static[int], order = colMajor): auto =
  constantMatrix(M, N, 0'f64, order)

proc zeros*(M, N: static[int], A: typedesc[float32], order = colMajor): auto =
  constantMatrix(M, N, 0'f32, order)

proc zeros*(M, N: static[int], A: typedesc[float64], order = colMajor): auto =
  constantMatrix(M, N, 0'f64, order)

proc ones*(M, N: static[int], order = colMajor): auto =
  constantMatrix(M, N, 1'f64, order)

proc ones*(M, N: static[int], A: typedesc[float32], order = colMajor): auto =
  constantMatrix(M, N, 1'f32, order)

proc ones*(M, N: static[int], A: typedesc[float64], order = colMajor): auto =
  constantMatrix(M, N, 1'f64, order)

proc eye*(N: static[int]): auto =
  dense.eye(N).asStatic(N, N)

proc eye*(N: static[int], A: typedesc[float32]): auto =
  dense.eye(N, A).asStatic(N, N)

proc eye*(N: static[int], A: typedesc[float64]): auto =
  dense.eye(N, A).asStatic(N, N)

proc randomMatrix*(M, N: static[int], max: float64 = 1): auto =
  dense.randomMatrix(M, N, max).asStatic(M, N)

proc randomMatrix*(M, N: static[int], max: float32): auto =
  dense.randomMatrix(M, N, max).asStatic(M, N)

# Accessors

proc len*[N: static[int]; A](v: StaticVector[N, A]): int {. inline .} = N

proc `[]`*[N: static[int]; A](v: StaticVector[N, A], i: int): A {. inline .} =
  dyn(v, A)[i]

proc `[]=`*[N: static[int]; A](v: var StaticVector[N, A], i: int, val: A) {. inline .} =
  dyn(v, A)[i] = val

proc dim*[M, N: static[int]; A](m: StaticMatrix[M, N, A]): tuple[rows, columns: int] {. inline .} =
  (rows: M, columns: N)

proc `[]`*[M, N: static[int]; A](m: StaticMatrix[M, N, A], i, j: int): A {. inline .} =
  dyn(m, A)[i, j]

proc `[]=`*[M, N: static[int]; A](m: var StaticMatrix[M, N, A], i, j: int, val: A) {. inline .} =
  dyn(m, A)[i, j] = val

proc column*[M, N: static[int]; A](m: StaticMatrix[M, N, A], j: int): StaticVector[M, A] {. inline .} =
  dyn(m, A).column(j).asStatic(M)

proc row*[M, N: static[int]; A](m: StaticMatrix[M, N, A], i: int): StaticVector[N, A] {. inline .} =
  dyn(m, A).row(i).asStatic(N)

# Iterators

iterator items*[N: static[int]; A](v: StaticVector[N, A]): auto {. inline .} =
  for i in 0 ..< N:
    yield v[i]

iterator pairs*[N: static[int]; A](v: StaticVector[N, A]): auto {. inline .} =
  for i in 0 ..< N:
    yield (i, v[i])

iterator items*[M, N: static[int]; A](m: StaticMatrix[M, N, A]): auto {. inline .} =
  for i in 0 ..< M:
    for j in 0 ..< N:
      yield m[i, j]

iterator pairs*[M, N: static[int]; A](m: StaticMatrix[M, N, A]): auto {. inline .} =
  for i in 0 ..< M:
    for j in 0 ..< N:
      yield ((i, j), m[i, j])

iterator columns*[M, N: static[int], A](m: StaticMatrix[M, N, A]): auto {. inline .} =
  for c in dyn(m, A).columns:
    yield c.asStatic(M)

iterator rows*[M, N: static[int], A](m: StaticMatrix[M, N, A]): auto {. inline .} =
  for r in dyn(m, A).rows:
    yield r.asStatic(N)

iterator columnsSlow*[M, N: static[int], A](m: StaticMatrix[M, N, A]): auto {. inline .} =
  for c in dyn(m, A).columnsSlow:
    yield c.asStatic(M)

iterator rowsSlow*[M, N: static[int], A](m: StaticMatrix[M, N, A]): auto {. inline .} =
  for r in dyn(m, A).rowsSlow:
    yield r.asStatic(N)

# Conversion

proc clone*[N: static[int]; A](v: StaticVector[N, A]): StaticVector[N, A] =
  dyn(v, A).clone().asStatic(N)

proc map*[N: static[int]; A, B](v: StaticVector[N, A], f: proc(x: A): B): auto =
  dyn(v, A).map(f).asStatic(N)

proc clone*[M, N: static[int]; A](m: StaticMatrix[M, N, A]): StaticMatrix[M, N, A] =
  dyn(m, A).clone().asStatic(M, N)

proc map*[M, N: static[int]; A, B](m: StaticMatrix[M, N, A], f: proc(x: A): B): auto =
  dyn(m, A).map(f).asStatic(M, N)

# Trivial operations

proc reshape*[M, N: static[int], T](m: StaticMatrix[M, N, T], A, B: static[int]): StaticMatrix[A, B, T] =
  static: doAssert(M * N == A * B, "The dimensions do not match: M = " & $(M) & ", N = " & $(N) & ", A = " & $(A) & ", B = " & $(B))
  dyn(m, T).reshape(A, B).asStatic(A, B)

proc asMatrix*[N: static[int], T](v: StaticVector[N, T], A, B: static[int], order: OrderType = colMajor): StaticMatrix[A, B, T] =
  static: doAssert(N == A * B, "The dimensions do not match: N = " & $(N) & ", A = " & $(A) & ", B = " & $(B))
  dyn(v, T).asMatrix(A, B, order).asStatic(A, B)

proc asVector*[M, N: static[int], A](m: StaticMatrix[M, N, A]): StaticVector[M * N, A] =
  dyn(m, A).asVector().asStatic(M * N)

proc t*[M, N: static[int], A](m: StaticMatrix[M, N, A]): StaticMatrix[N, M, A] =
  dyn(m, A).t().asStatic(N, M)

proc T*[M, N: static[int], A](m: StaticMatrix[M, N, A]): StaticMatrix[N, M, A] =
  dyn(m, A).T().asStatic(N, M)

# Collection

proc cumsum*[N: static[int]; A: SomeFloat](v: StaticVector[N, A]): StaticVector[N, A] =
  dyn(v, A).cumsum().asStatic(N)

proc sum*[N: static[int]; A: SomeFloat](v: StaticVector[N, A]): A =
  dyn(v, A).sum()

proc mean*[N: static[int]; A: SomeFloat](v: StaticVector[N, A]): A =
  dyn(v, A).mean()

proc variance*[N: static[int]; A: SomeFloat](v: StaticVector[N, A]): A =
  dyn(v, A).variance()

proc stddev*[N: static[int]; A: SomeFloat](v: StaticVector[N, A]): A =
  dyn(v, A).stddev()

proc sum*[M, N: static[int]; A: SomeFloat](m: StaticMatrix[M, N, A]): A =
  dyn(m, A).sum()

proc mean*[M, N: static[int]; A: SomeFloat](m: StaticMatrix[M, N, A]): A =
  dyn(m, A).mean()

proc variance*[M, N: static[int]; A: SomeFloat](m: StaticMatrix[M, N, A]): A =
  dyn(m, A).variance()

proc stddev*[M, N: static[int]; A: SomeFloat](m: StaticMatrix[M, N, A]): A =
  dyn(m, A).stddev()

# Operations

# BLAS level 1 operations

proc `*=`*[N: static[int], A: SomeFloat](v: var StaticVector[N, A], k: A) {. inline .} =
  dyn(v, A) *= k

proc `*`*[N: static[int], A: SomeFloat](v: StaticVector[N, A], k: float64): StaticVector[N, A]  {. inline .} =
  (dyn(v, A) * k).asStatic(N)

proc `+=`*[N: static[int], A: SomeFloat](v: var StaticVector[N, A], w: StaticVector[N, A]) {. inline .} =
  dyn(v, A) += dyn(w, A)

proc `+`*[N: static[int], A: SomeFloat](v, w: StaticVector[N, A]): StaticVector[N, A]  {. inline .} =
  (dyn(v, A) + dyn(w, A)).asStatic(N)

proc `-=`*[N: static[int], A: SomeFloat](v: var StaticVector[N, A], w: StaticVector[N, A]) {. inline .} =
  dyn(v, A) -= dyn(w, A)

proc `-`*[N: static[int], A: SomeFloat](v, w: StaticVector[N, A]): StaticVector[N, A]  {. inline .} =
  (dyn(v, A) - dyn(w, A)).asStatic(N)

proc `*`*[N: static[int], A: SomeFloat](v, w: StaticVector[N, A]): float64 {. inline .} =
  dyn(v, A) * dyn(w, A)

proc l_2*[N: static[int], A: SomeFloat](v: StaticVector[N, A]): auto {. inline .} =
  l_2(dyn(v, A))

proc l_1*[N: static[int], A: SomeFloat](v: StaticVector[N, A]): auto {. inline .} =
  l_1(dyn(v, A))

proc maxIndex*[N: static[int], A: SomeFloat](v: StaticVector[N, A]): tuple[i: int, val: A] =
  maxIndex(dyn(v, A))

template max*(v: StaticVector): auto = maxIndex(v).val

proc minIndex*[N: static[int], A: SomeFloat](v: StaticVector[N, A]): tuple[i: int, val: A] =
  minIndex(dyn(v, A))

template min*(v: StaticVector): auto = minIndex(v).val

proc `*`*[M, N: static[int], A: SomeFloat](m: StaticMatrix[M, N, A], v: StaticVector[N, A]): StaticVector[M, A]  {. inline .} =
  (dyn(m, A) * dyn(v, A)).asStatic(M)

proc `*=`*[M, N: static[int], A: SomeFloat](m: var StaticMatrix[M, N, A], k: float64) {. inline .} =
  dyn(m, A) *= k

proc `*`*[M, N: static[int], A: SomeFloat](m: StaticMatrix[M, N, A], k: float64): StaticMatrix[M, N, A]  {. inline .} =
  (dyn(m, A) * k).asStatic(M, N)

template `*`*[A: SomeFloat](k: A, v: StaticVector or StaticMatrix): auto = v * k

template `/`*[A: SomeFloat](v: StaticVector or StaticMatrix, k: A): auto = v * (1 / k)

template `/=`*[A: SomeFloat](v: var StaticVector or var StaticMatrix, k: A) =
  v *= (1 / k)

proc `+=`*[M, N: static[int], A: SomeFloat](a: var StaticMatrix[M, N, A], b: StaticMatrix[M, N, A]) {. inline .} =
  dyn(a, A) += dyn(b, A)

proc `+`*[M, N: static[int], A: SomeFloat](a, b: StaticMatrix[M, N, A]): StaticMatrix[M, N, A]  {. inline .} =
  (dyn(a, A) + dyn(b, A)).asStatic(M, N)

proc `-=`*[M, N: static[int], A: SomeFloat](a: var StaticMatrix[M, N, A], b: StaticMatrix[M, N, A]) {. inline .} =
  dyn(a, A) -= dyn(b, A)

proc `-`*[M, N: static[int], A: SomeFloat](a, b: StaticMatrix[M, N, A]): StaticMatrix[M, N, A]  {. inline .} =
  (dyn(a, A) - dyn(b, A)).asStatic(M, N)

proc l_2*[M, N: static[int], A: SomeFloat](m: StaticMatrix[M, N, A]): auto {. inline .} =
  l_2(dyn(m, A))

proc l_1*[M, N: static[int], A: SomeFloat](m: StaticMatrix[M, N, A]): auto {. inline .} =
  l_1(dyn(m, A))

proc max*[M, N: static[int], A: SomeFloat](m: StaticMatrix[M, N, A]): A =
  max(dyn(m, A))

proc min*[M, N: static[int], A: SomeFloat](m: StaticMatrix[M, N, A]): A =
  min(dyn(m, A))

# BLAS level 3 operations

proc `*`*[M, N, K: static[int]; A: SomeFloat](
  m: StaticMatrix[M, K, A],
  n: StaticMatrix[K, N, A]
): StaticMatrix[M, N, A] = (dyn(m, A) * dyn(n, A)).asStatic(M, N)

# Hadamard (component-wise) product

proc `|*|`*[N: static[int], A: SomeFloat](a, b: StaticVector[N, A]): StaticVector[N, A] =
  (dyn(a, A) |*| dyn(b, A)).asStatic(N)

proc `|*|`*[M, N: static[int], A: SomeFloat](a, b: StaticMatrix[M, N, A]): StaticMatrix[M, N, A] =
  (dyn(a, A) |*| dyn(b, A)).asStatic(M, N)