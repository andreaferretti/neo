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
import nimblas, sequtils, random

type
  Vector*[A] = seq[A]
  Matrix*[A] = ref object
    order*: OrderType
    M*, N*: int
    data*: seq[A]

# Float pointers
template fp[A](v: Vector[A]): ptr A = cast[ptr A](unsafeAddr(v[0]))

template fp[A](m: Matrix[A]): ptr A = cast[ptr A](unsafeAddr(m.data[0]))

# Equality

template elem(m, i, j: untyped): auto =
  if m.order == colMajor: m.data[j * m.M + i]
  else: m.data[i * m.N + j]

proc slowEq(m, n: Matrix): bool =
  if m.M != n.M or m.N != n.N:
    return false
  for i in 0 ..< m.M:
    for j in 0 ..< m.N:
      if elem(m, i, j) != elem(n, i, j):
        return false
  return true

proc `==`*[A](m, n: Matrix[A]): bool =
  if m.order == n.order: m.data == n.data
  elif m.order == colMajor: slowEq(m, n)
  else: slowEq(m, n)

# Conversion

proc to32*(v: Vector[float64]): Vector[float32] = v.mapIt(it.float32)

proc to64*(v: Vector[float32]): Vector[float64] = v.mapIt(it.float64)

proc to32*(m: Matrix[float64]): Matrix[float32] =
  result = Matrix[float32](data:m.data.mapIt(it.float32), order:m.order, M:m.M, N:m.N)

proc to64*(m: Matrix[float32]): Matrix[float64] =
  result = Matrix[float64](data:m.data.mapIt(it.float64), order:m.order, M:m.M, N:m.N)

proc makeVector*[A](N: int, f: proc (i: int): A): Vector[A] =
  result = newSeq[A](N)
  for i in 0 ..< N:
    result[i] = f(i)

template makeVectorI*[A](N: int, f: untyped): Vector[A] =
  var result = newSeq[A](N)
  for i {.inject.} in 0 ..< N:
    result[i] = f
  result

proc randomVector*(N: int, max: float64 = 1): Vector[float64] =
  makeVectorI[float64](N, random(max))

proc randomVector*(N: int, max: float32): Vector[float32] =
  makeVectorI[float32](N, random(max).float32)

proc constantVector*[A](N: int, a: A): Vector[A] = makeVectorI[A](N, a)

proc zeros*(N: int): auto = constantVector(N, 0'f64)

proc zeros*(N: int, A: typedesc[float32]): auto = constantVector(N, 0'f32)

proc zeros*(N: int, A: typedesc[float64]): auto = constantVector(N, 0'f64)

proc ones*(N: int): auto = constantVector(N, 1'f64)

proc ones*(N: int, A: typedesc[float32]): auto = constantVector(N, 1'f32)

proc ones*(N: int, A: typedesc[float64]): auto = constantVector(N, 1'f64)

proc makeMatrix*[A](M, N: int, f: proc (i, j: int): A, order: OrderType): Matrix[A] =
  new result
  result.data = newSeq[A](M * N)
  result.M = M
  result.N = N
  result.order = order
  if order == colMajor:
    for i in 0 ..< M:
      for j in 0 ..< N:
        result.data[j * M + i] = f(i, j)
  else:
    for i in 0 ..< M:
      for j in 0 ..< N:
        result.data[i * N + j] = f(i, j)

template makeMatrixIJ*[A](M1, N1: int, f: untyped, ord = colMajor): auto =
  new result
  result.data = newSeq[A](M1 * N1)
  result.M = M1
  result.N = N1
  result.order = order
  if ord == colMajor:
    for i {.inject.} in 0 ..< M1:
      for j {.inject.} in 0 ..< N1:
        result.data[j * M1 + i] = f
  else:
    for i {.inject.} in 0 ..< M1:
      for j {.inject.} in 0 ..< N1:
        result.data[i * N1 + j] = f
  result

proc randomMatrix*(M, N: int, max: float64 = 1, order = colMajor): Matrix[float64] =
  makeMatrixIJ[float64](M, N, random(max), order)

proc randomMatrix*(M, N: int, max: float32 = 1, order = colMajor): Matrix[float32] =
  makeMatrixIJ[float32](M, N, random(max).float32, order)

proc constantMatrix*[A](M, N: int, a: A, order = colMajor): Matrix[A] =
  makeMatrixIJ[A](M, N, a, order)

proc zeros*(M, N: int): auto = constantMatrix(M, N, 0'f64)

proc zeros*(M, N: int, A: typedesc[float32]): auto = constantMatrix(M, N, 0'f32)

proc zeros*(M, N: int, A: typedesc[float64]): auto = constantMatrix(M, N, 0'f64)

proc ones*(M, N: int): auto = constantMatrix(M, N, 1'f64)

proc ones*(M, N: int, A: typedesc[float32]): auto = constantMatrix(M, N, 1'f32)

proc ones*(M, N: int, A: typedesc[float64]): auto = constantMatrix(M, N, 1'f64)

proc eye*(N: int, order = colMajor): Matrix[float64] =
  makeMatrixIJ[float64](N, N, if i == j: 1 else: 0, order)

proc eye*(N: int, A: typedesc[float32], order = colMajor): Matrix[float32] =
  # should use makeMatrixIJ, go figure...
  makeMatrix[float32](N, N, proc(i, j: int): float32 = (if i == j: 1 else: 0), order)

proc matrix*[A](xs: seq[seq[A]], order = colMajor): Matrix[A] =
  # should use makeMatrixIJ, go figure...
  makeMatrixIJ[A](xs.len, xs[0].len, proc(i, j: int): float32 = xs[i][j], order)

proc `[]`*[A](m: Matrix[A], i, j: int): A {. inline .} =
  if m.order == colMajor: m.data[j * m.M + i]
  else: m.data[i * m.N + j]

proc `[]=`*[A](m: var Matrix[A], i, j: int, val: A) {. inline .} =
  if m.order == colMajor:
    m.data[j * m.M + i] = val
  else:
    m.data[i * m.N + j] = val

proc column*[A](m: Matrix[A], j: int): Vector[A] {. inline .} =
  result = newSeq[A](m.M)
  for i in 0 .. < m.M:
    result[i] = m[i, j]

proc row*[A](m: Matrix[A], i: int): Vector[A] {. inline .} =
  result = newSeq[A](m.N)
  for j in 0 .. < m.N:
    result[j] = m[i, j]

proc dim*(m: Matrix): tuple[rows, columns: int] = (m.M, m.N)

proc clone*[A](m: Matrix[A]): Matrix[A] =
  Matrix[A](data: m.data, order: m.order, M: m.M, N: m.N)

proc map*[A](m: Matrix[A], f: proc(x: A): A): Matrix[A] =
  result = Matrix[A](data: newSeq[A](m.M * m.N), order: m.order, M: m.M, N: m.N)
  for i in 0 ..< (m.M * m.N):
    result.data[i] = f(m.data[i])

iterator columns*[A](m: Matrix[A]): auto {. inline .} =
  for i in 0 ..< m.N:
    yield m.column(i)

iterator rows*[A](m: Matrix[A]): auto {. inline .} =
  for i in 0 ..< m.M:
    yield m.row(i)

iterator items*[A](m: Matrix[A]): auto {. inline .} =
  for i in 0 ..< m.M:
    for j in 0 ..< m.N:
      yield m[i, j]

iterator pairs*[A](m: Matrix[A]): auto {. inline .} =
  for i in 0 ..< m.M:
    for j in 0 ..< m.N:
      yield ((i, j), m[i, j])

proc toStringHorizontal[A](v: Vector[A]): string =
  result = "[ "
  for i in 0 ..< (v.len - 1):
    result &= $(v[i]) & "\t"
  result &= $(v[v.len - 1]) & " ]"

proc `$`*[A](m: Matrix[A]): string =
  result = "[ "
  for i in 0 .. < (m.M - 1):
    result &= toStringHorizontal(m.row(i)) & "\n  "
  result &= toStringHorizontal(m.row(m.M - 1)) & " ]"