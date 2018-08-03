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

template `*`*[A: SomeFloat](k: A, v: StaticVector or StaticMatrix): auto = v * k

template `/`*[A: SomeFloat](v: StaticVector or StaticMatrix, k: A): auto = v * (1 / k)

template `/=`*[A: SomeFloat](v: var StaticVector or var StaticMatrix, k: A) =
  v *= (1 / k)

# proc `*`*[N: static[int], A: SomeFloat](v, w: StaticVector[N, A]): float64 {. inline .} = dot(N, v.fp, 1, w.fp, 1)

# proc `*`*(v, w: DVector32): float32 {. inline .} =
#   assert(v.len == w.len)
#   let N = v.len
#   dot(N, v.fp, 1, w.fp, 1)

# proc `*`*(v, w: DVector64): float64 {. inline .} =
#   assert(v.len == w.len)
#   let N = v.len
#   dot(N, v.fp, 1, w.fp, 1)

# proc l_2*[N: static[int], A: SomeFloat](v: Vector32[N] or StaticVector[N, A]): auto {. inline .} = nrm2(N, v.fp, 1)

# proc l_2*(v: DVector32 or DVector64): auto {. inline .} = nrm2(v.len, v.fp, 1)

# proc l_1*[N: static[int], A: SomeFloat](v: Vector32[N] or StaticVector[N, A]): auto {. inline .} = asum(N, v.fp, 1)

# proc l_1*(v: DVector32 or DVector64): auto {. inline .} = asum(v.len, v.fp, 1)

# template maxIndexPrivate(N, v: untyped): auto =
#   var
#     j = 0
#     m = v[0]
#   for i, val in v:
#     if val > m:
#       j = i
#       m = val
#   (j, m)

# proc maxIndex*[N: static[int], A: SomeFloat](v: Vector32[N]): tuple[i: int, val: float32] =
#   maxIndexPrivate(N, v)

# proc maxIndex*[N: static[int], A: SomeFloat](v: StaticVector[N, A]): tuple[i: int, val: float64] =
#   maxIndexPrivate(N, v)

# proc maxIndex*(v: DVector32): tuple[i: int, val: float32] =
#   maxIndexPrivate(v.len, v)

# proc maxIndex*(v: DVector64): tuple[i: int, val: float64] =
#   maxIndexPrivate(v.len, v)

# template max*(v: Vector32 or Vector64): auto = maxIndex(v).val

# template minIndexPrivate(N, v: untyped): auto =
#   var
#     j = 0
#     m = v[0]
#   for i, val in v:
#     if val < m:
#       j = i
#       m = val
#   return (j, m)

# proc minIndex*[N: static[int], A: SomeFloat](v: Vector32[N]): tuple[i: int, val: float32] =
#   minIndexPrivate(N, v)

# proc minIndex*[N: static[int], A: SomeFloat](v: StaticVector[N, A]): tuple[i: int, val: float64] =
#   minIndexPrivate(N, v)

# proc minIndex*(v: DVector32): tuple[i: int, val: float32] =
#   minIndexPrivate(v.len, v)

# proc minIndex*(v: DVector64): tuple[i: int, val: float64] =
#   minIndexPrivate(v.len, v)

# template min*(v: Vector32 or Vector64): auto = minIndex(v).val

# proc `*`*[M, N: static[int]](a: Matrix32[M, N], v: Vector32[N]): Vector32[M]  {. inline .} =
#   new result
#   let lda = if a.order == colMajor: M.int else: N.int
#   gemv(a.order, noTranspose, M, N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

# proc `*`*(a: DMatrix32, v: DVector32): DVector32  {. inline .} =
#   assert(a.N == v.len)
#   result = newSeq[float32](a.M)
#   let lda = if a.order == colMajor: a.M.int else: a.N.int
#   gemv(a.order, noTranspose, a.M, a.N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

# proc `*`*[M, N: static[int]](a: Matrix64[M, N], v: StaticVector[N, A]): Vector64[M]  {. inline .} =
#   new result
#   let lda = if a.order == colMajor: M.int else: N.int
#   gemv(a.order, noTranspose, M, N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

# proc `*`*(a: DMatrix64, v: DVector64): DVector64  {. inline .} =
#   assert(a.N == v.len)
#   result = newSeq[float64](a.M)
#   let lda = if a.order == colMajor: a.M.int else: a.N.int
#   gemv(a.order, noTranspose, a.M, a.N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

# proc `*=`*[M, N: static[int]](m: var Matrix32[M, N], k: float32) {. inline .} = scal(M * N, k, m.fp, 1)

# proc `*`*[M, N: static[int]](m: Matrix32[M, N], k: float32): Matrix32[M, N]  {. inline .} =
#   new result.data
#   result.order = m.order
#   copy(M * N, m.fp, 1, result.fp, 1)
#   scal(M * N, k, result.fp, 1)

# proc `*=`*(m: var DMatrix32, k: float32) {. inline .} = scal(m.M * m.N, k, m.fp, 1)

# proc `*`*(m: DMatrix32, k: float32): DMatrix32  {. inline .} =
#   result.initLike(m)
#   copy(m.len, m.fp, 1, result.fp, 1)
#   scal(m.len, k, result.fp, 1)

# proc `*=`*[M, N: static[int]](m: var Matrix64[M, N], k: float64) {. inline .} = scal(M * N, k, m.fp, 1)

# proc `*`*[M, N: static[int]](m: Matrix64[M, N], k: float64): Matrix64[M, N]  {. inline .} =
#   new result.data
#   result.order = m.order
#   copy(M * N, m.fp, 1, result.fp, 1)
#   scal(M * N, k, result.fp, 1)

# proc `*=`*(m: var DMatrix64, k: float64) {. inline .} = scal(m.M * m.N, k, m.fp, 1)

# proc `*`*(m: DMatrix64, k: float64): DMatrix64  {. inline .} =
#   result.initLike(m)
#   copy(m.len, m.fp, 1, result.fp, 1)
#   scal(m.len, k, result.fp, 1)

# template `*`*(k: float32, v: Vector32 or Matrix32 or DVector32 or DMatrix32): untyped = v * k

# template `/`*(v: Vector32 or Matrix32 or DVector32 or DMatrix32, k: float32): untyped = v * (1 / k)

# template `/=`*(v: var Vector32 or var Matrix32 or var DVector32 or var DMatrix32, k: float32): untyped = v *= (1 / k)

# template `*`*(k: float64, v: Vector64 or Matrix64 or DVector64 or DMatrix64): untyped = v * k

# template `/`*(v: Vector64 or Matrix64 or DVector64 or DMatrix64, k: float64): untyped = v * (1 / k)

# template `/=`*(v: var Vector64 or var Matrix64 or var DVector64 or var DMatrix64, k: float64): untyped = v *= (1 / k)

# template matrixAdd(M, N, a, b: untyped, A: typedesc) =
#   if a.order == b.order:
#     axpy(M * N, 1, b.fp, 1, a.fp, 1)
#   elif a.order == colMajor and b.order == rowMajor:
#     let
#       a_data = cast[ref array[N, array[M, A]]](a.data)
#       b_data = cast[ref array[M, array[N, A]]](b.data)
#     for i in 0 .. < M:
#       for j in 0 .. < N:
#         a_data[j][i] += b_data[i][j]
#   else:
#     let
#       a_data = cast[ref array[M, array[N, A]]](a.data)
#       b_data = cast[ref array[N, array[M, A]]](b.data)
#     for i in 0 .. < M:
#       for j in 0 .. < N:
#         a_data[i][j] += b_data[j][i]

# template matrixDAdd(a, b: untyped) =
#   assert a.M == b.M and a.N == a.N
#   if a.order == b.order:
#     axpy(a.M * a.N, 1, b.fp, 1, a.fp, 1)
#   elif a.order == colMajor and b.order == rowMajor:
#     for i in 0 .. < a.M:
#       for j in 0 .. < a.N:
#         a.data[j * a.M + i] += b.data[i * b.N + j]
#   else:
#     for i in 0 .. < a.M:
#       for j in 0 .. < a.N:
#         a.data[i * a.N + j] += b.data[j * b.M + i]

# proc `+=`*[M, N: static[int]](a: var Matrix32[M, N], b: Matrix32[M, N]) {. inline .} =
#   matrixAdd(M, N, a, b, float32)

# proc `+=`*(a: var DMatrix32, b: DMatrix32) {. inline .} = matrixDAdd(a, b)

# proc `+=`*[M, N: static[int]](a: var Matrix64[M, N], b: Matrix64[M, N]) {. inline .} =
#   matrixAdd(M, N, a, b, float64)

# proc `+=`*(a: var DMatrix64, b: DMatrix64) {. inline .} = matrixDAdd(a, b)

# proc `+`*[M, N: static[int]](a, b: Matrix32[M, N]): Matrix32[M, N] {. inline .} =
#   new result.data
#   result.order = a.order
#   copy(M * N, a.fp, 1, result.fp, 1)
#   result += b

# proc `+`*(a, b: DMatrix32): DMatrix32 {. inline .} =
#   result.initLike(a)
#   copy(a.len, a.fp, 1, result.fp, 1)
#   result += b

# proc `+`*[M, N: static[int]](a, b: Matrix64[M, N]): Matrix64[M, N]  {. inline .} =
#   new result.data
#   result.order = a.order
#   copy(M * N, a.fp, 1, result.fp, 1)
#   result += b

# proc `+`*(a, b: DMatrix64): DMatrix64 {. inline .} =
#   result.initLike(a)
#   copy(a.len, a.fp, 1, result.fp, 1)
#   result += b

# template matrixSub(M, N, a, b: untyped, A: typedesc) =
#   if a.order == b.order:
#     axpy(M * N, -1, b.fp, 1, a.fp, 1)
#   elif a.order == colMajor and b.order == rowMajor:
#     let
#       a_data = cast[ref array[N, array[M, A]]](a.data)
#       b_data = cast[ref array[M, array[N, A]]](b.data)
#     for i in 0 .. < M:
#       for j in 0 .. < N:
#         a_data[j][i] -= b_data[i][j]
#   else:
#     let
#       a_data = cast[ref array[M, array[N, A]]](a.data)
#       b_data = cast[ref array[N, array[M, A]]](b.data)
#     for i in 0 .. < M:
#       for j in 0 .. < N:
#         a_data[i][j] -= b_data[j][i]

# template matrixDSub(a, b: untyped) =
#   assert a.M == b.M and a.N == a.N
#   if a.order == b.order:
#     axpy(a.M * a.N, -1, b.fp, 1, a.fp, 1)
#   elif a.order == colMajor and b.order == rowMajor:
#     for i in 0 .. < a.M:
#       for j in 0 .. < a.N:
#         a.data[j * a.M + i] -= b.data[i * b.N + j]
#   else:
#     for i in 0 .. < a.M:
#       for j in 0 .. < a.N:
#         a.data[i * a.N + j] -= b.data[j * b.M + i]

# proc `-=`*[M, N: static[int]](a: var Matrix32[M, N], b: Matrix32[M, N]) {. inline .} =
#   matrixSub(M, N, a, b, float32)

# proc `-=`*(a: var DMatrix32, b: DMatrix32) {. inline .} = matrixDSub(a, b)

# proc `-`*[M, N: static[int]](a, b: Matrix32[M, N]): Matrix32[M, N]  {. inline .} =
#   new result.data
#   result.order = a.order
#   copy(M * N, a.fp, 1, result.fp, 1)
#   result -= b

# proc `-`*(a, b: DMatrix32): DMatrix32 {. inline .} =
#   result.initLike(a)
#   copy(a.len, a.fp, 1, result.fp, 1)
#   result -= b

# proc `-=`*[M, N: static[int]](a: var Matrix64[M, N], b: Matrix64[M, N]) {. inline .} =
#   matrixSub(M, N, a, b, float64)

# proc `-=`*(a: var DMatrix64, b: DMatrix64) {. inline .} = matrixDSub(a, b)

# proc `-`*[M, N: static[int]](a, b: Matrix64[M, N]): Matrix64[M, N]  {. inline .} =
#   new result.data
#   result.order = a.order
#   copy(M * N, a.fp, 1, result.fp, 1)
#   result -= b

# proc `-`*(a, b: DMatrix64): DMatrix64 {. inline .} =
#   result.initLike(a)
#   copy(a.len, a.fp, 1, result.fp, 1)
#   result -= b

# proc l_2*[M, N: static[int]](m: Matrix32[M, N] or Matrix64[M, N]): auto {. inline .} =
#   nrm2(M * N, m.fp, 1)

# proc l_2*(m: DMatrix32 or DMatrix64): auto {. inline .} = nrm2(m.len, m.fp, 1)

# proc l_1*[M, N: static[int]](m: Matrix32[M, N] or Matrix64[M, N]): auto {. inline .} =
#   asum(M * N, m.fp, 1)

# proc l_1*(m: DMatrix32 or DMatrix64): auto {. inline .} = asum(m.len, m.fp, 1)

# template max*(m: Matrix32 or Matrix64 or DMatrix32 or DMatrix64): auto = max(m.data)

# template min*(m: Matrix32 or Matrix64 or DMatrix32 or DMatrix64): auto = min(m.data)

# template matrixMult(M, N, K, a, b, result: untyped): auto =
#   new result.data
#   if a.order == colMajor and b.order == colMajor:
#     result.order = colMajor
#     gemm(colMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, M, b.fp, K, 0, result.fp, M)
#   elif a.order == rowMajor and b.order == rowMajor:
#     result.order = rowMajor
#     gemm(rowMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, K, b.fp, N, 0, result.fp, N)
#   elif a.order == colMajor and b.order == rowMajor:
#     result.order = colMajor
#     gemm(colMajor, noTranspose, transpose, M, N, K, 1, a.fp, M, b.fp, N, 0, result.fp, M)
#   else:
#     result.order = colMajor
#     gemm(colMajor, transpose, noTranspose, M, N, K, 1, a.fp, K, b.fp, K, 0, result.fp, M)

# template matrixMultD(a, b, result: untyped): auto =
#   new result
#   let
#     M = a.M
#     K = a.N
#     N = b.N
#   assert b.M == K
#   when a is DMatrix32:
#     result.data = newSeq[float32](M * N)
#   when a is DMatrix64:
#     result.data = newSeq[float64](M * N)
#   result.M = M
#   result.N = N
#   if a.order == colMajor and b.order == colMajor:
#     result.order = colMajor
#     gemm(colMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, M, b.fp, K, 0, result.fp, M)
#   elif a.order == rowMajor and b.order == rowMajor:
#     result.order = rowMajor
#     gemm(rowMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, K, b.fp, N, 0, result.fp, N)
#   elif a.order == colMajor and b.order == rowMajor:
#     result.order = colMajor
#     gemm(colMajor, noTranspose, transpose, M, N, K, 1, a.fp, M, b.fp, N, 0, result.fp, M)
#   else:
#     result.order = colMajor
#     gemm(colMajor, transpose, noTranspose, M, N, K, 1, a.fp, K, b.fp, K, 0, result.fp, M)

# proc `*`*[M, N, K: static[int]](a: Matrix32[M, K], b: Matrix32[K, N]): Matrix32[M, N] {. inline .} =
#   matrixMult(M, N, K, a, b, result)

# proc `*`*(a: DMatrix32, b: DMatrix32): DMatrix32 {. inline .} = matrixMultD(a, b, result)

# proc `*`*[M, N, K: static[int]](a: Matrix64[M, K], b: Matrix64[K, N]): Matrix64[M, N] {. inline .} =
#   matrixMult(M, N, K, a, b, result)

# proc `*`*(a: DMatrix64, b: DMatrix64): DMatrix64 {. inline .} = matrixMultD(a, b, result)

# proc compareApprox(a, b: Vector32 or Vector64 or DVector32 or DVector64 or Matrix32 or Matrix64 or DMatrix32 or DMatrix64): bool =
#   const epsilon = 0.000001
#   let
#     aNorm = l_1(a)
#     bNorm = l_1(b)
#     dNorm = l_1(a - b)
#   return (dNorm / (aNorm + bNorm)) < epsilon

# template `=~`*[N: static[int], A: SomeFloat](v, w: Vector32[N]): bool = compareApprox(v, w)

# template `=~`*[N: static[int], A: SomeFloat](v, w: StaticVector[N, A]): bool = compareApprox(v, w)

# template `=~`*(v, w: DVector32): bool = compareApprox(v, w)

# template `=~`*(v, w: DVector64): bool = compareApprox(v, w)

# proc `=~`*[M, N: static[int]](m, n: Matrix32[M, N]): bool = compareApprox(m, n)

# proc `=~`*[M, N: static[int]](m, n: Matrix64[M, N]): bool = compareApprox(m, n)

# template `=~`*(v, w: DMatrix32): bool = compareApprox(v, w)

# template `=~`*(v, w: DMatrix64): bool = compareApprox(v, w)

# template `!=~`*(a, b: Vector32 or Vector64 or DVector32 or DVector64 or Matrix32 or Matrix64 or DMatrix32 or DMatrix64): bool =
#   not (a =~ b)

# # Hadamard (component-wise) product
# template hadamardV(N, a, b: untyped, T: typedesc): auto =
#   var x = zeros(N, T)
#   for i in 0 .. < N:
#     x[i] = a[i] * b[i]
#   x

# template hadamardM(M, N, a, b: untyped, T: typedesc): auto =
#   var x = zeros(M, N, T)
#   if a.order == b.order:
#     x.order = a.order
#     for i in 0 .. < (M * N):
#       x.data[i] = a.data[i] * b.data[i]
#   else:
#     for i in 0 .. < M:
#       for j in 0 .. < N:
#         x[i, j] = a[i, j] * b[i, j]
#   x

# proc `|*|`*[N: static[int], A: SomeFloat](a, b: Vector32[N]): Vector32[N] =
#   hadamardV(N, a, b, float32)

# proc `|*|`*[N: static[int], A: SomeFloat](a, b: StaticVector[N, A]): StaticVector[N, A] =
#   hadamardV(N, a, b, float64)

# proc `|*|`*(a, b: DVector32): DVector32 =
#   assert a.len == b.len
#   hadamardV(a.len, a, b, float32)

# proc `|*|`*(a, b: DVector64): DVector64 =
#   assert a.len == b.len
#   hadamardV(a.len, a, b, float64)

# proc `|*|`*[M, N: static[int]](a, b: Matrix32[M, N]): Matrix32[M, N] =
#   # hadamardM(M, N, a, b, float32)
#   result = zeros(M, N, float32)
#   if a.order == b.order:
#     result.order = a.order
#     for i in 0 .. < (M * N):
#       result.data[i] = a.data[i] * b.data[i]
#   else:
#     for i in 0 .. < M:
#       for j in 0 .. < N:
#         result[i, j] = a[i, j] * b[i, j]

# proc `|*|`*[M, N: static[int]](a, b: Matrix64[M, N]): Matrix64[M, N] =
#   # hadamardM(M, N, a, b, float64)
#   result = zeros(M, N)
#   if a.order == b.order:
#     result.order = a.order
#     for i in 0 .. < (M * N):
#       result.data[i] = a.data[i] * b.data[i]
#   else:
#     for i in 0 .. < M:
#       for j in 0 .. < N:
#         result[i, j] = a[i, j] * b[i, j]

# proc `|*|`*(a, b: DMatrix32): DMatrix32 =
#   assert a.dim == b.dim
#   let (m, n) = a.dim
#   return hadamardM(m, n, a, b, float32)

# proc `|*|`*(a, b: DMatrix64): DMatrix64 =
#   assert a.dim == b.dim
#   let (m, n) = a.dim
#   return hadamardM(m, n, a, b, float64)

# BLAS level 3 operations

proc `*`*[M, N, K: static[int]; A: SomeFloat](
  m: StaticMatrix[M, K, A],
  n: StaticMatrix[K, N, A]
): StaticMatrix[M, N, A] = (dyn(m, A) * dyn(n, A)).asStatic(M, N)