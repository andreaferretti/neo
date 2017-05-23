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
import nimblas, sequtils, random, math

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

# Initializers

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

proc makeMatrix*[A](M, N: int, f: proc (i, j: int): A, order = colMajor): Matrix[A] =
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

template makeMatrixIJ*(A: typedesc, M1, N1: int, f: untyped, ord = colMajor): auto =
  var r: Matrix[A]
  new(r)
  r.data = newSeq[A](M1 * N1)
  r.M = M1
  r.N = N1
  r.order = ord
  if ord == colMajor:
    for i {.inject.} in 0 ..< M1:
      for j {.inject.} in 0 ..< N1:
        r.data[j * M1 + i] = f
  else:
    for i {.inject.} in 0 ..< M1:
      for j {.inject.} in 0 ..< N1:
        r.data[i * N1 + j] = f
  r

proc randomMatrix*[A: SomeReal](M, N: int, max: A = 1, order = colMajor): Matrix[A] =
  new(result)
  result.data = newSeq[A](M * N)
  result.M = M
  result.N = N
  result.order = order
  for i in 0 ..< (M * N):
    result.data[i] = random(max)

proc randomMatrix*(M, N: int, order = colMajor): Matrix[float64] =
  randomMatrix(M, N, 1'f64, order)

proc constantMatrix*[A](M, N: int, a: A, order = colMajor): Matrix[A] =
  new(result)
  result.data = sequtils.repeat(a, M * N)
  result.M = M
  result.N = N
  result.order = order

proc zeros*(M, N: int): auto = constantMatrix(M, N, 0'f64)

proc zeros*(M, N: int, A: typedesc[float32]): auto = constantMatrix(M, N, 0'f32)

proc zeros*(M, N: int, A: typedesc[float64]): auto = constantMatrix(M, N, 0'f64)

proc ones*(M, N: int): auto = constantMatrix(M, N, 1'f64)

proc ones*(M, N: int, A: typedesc[float32]): auto = constantMatrix(M, N, 1'f32)

proc ones*(M, N: int, A: typedesc[float64]): auto = constantMatrix(M, N, 1'f64)

proc eye*(N: int, order = colMajor): Matrix[float64] =
  makeMatrixIJ(float64, N, N, if i == j: 1 else: 0, order)

proc eye*(N: int, A: typedesc[float32], order = colMajor): Matrix[float32] =
  makeMatrixIJ(float32, N, N, if i == j: 1 else: 0, order)

proc matrix*[A](xs: seq[seq[A]], order = colMajor): Matrix[A] =
  makeMatrixIJ(A, xs.len, xs[0].len, xs[i][j], order)

# Accessors

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
  for i in 0 ..< m.M:
    result[i] = m[i, j]

proc row*[A](m: Matrix[A], i: int): Vector[A] {. inline .} =
  result = newSeq[A](m.N)
  for j in 0 ..< m.N:
    result[j] = m[i, j]

proc dim*(m: Matrix): tuple[rows, columns: int] = (m.M, m.N)

proc clone*[A](m: Matrix[A]): Matrix[A] =
  Matrix[A](data: m.data, order: m.order, M: m.M, N: m.N)

proc map*[A](m: Matrix[A], f: proc(x: A): A): Matrix[A] =
  result = Matrix[A](data: newSeq[A](m.M * m.N), order: m.order, M: m.M, N: m.N)
  for i in 0 ..< (m.M * m.N):
    result.data[i] = f(m.data[i])

# Iterators

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

# Pretty printing

proc toStringHorizontal[A](v: Vector[A]): string =
  result = "[ "
  for i in 0 ..< (v.len - 1):
    result &= $(v[i]) & "\t"
  result &= $(v[v.len - 1]) & " ]"

proc `$`*[A](m: Matrix[A]): string =
  result = "[ "
  for i in 0 ..< (m.M - 1):
    result &= toStringHorizontal(m.row(i)) & "\n  "
  result &= toStringHorizontal(m.row(m.M - 1)) & " ]"

# Trivial operations

proc t*[A](m: Matrix[A]): Matrix[A] =
  new result
  result.M = m.N
  result.N = m.M
  result.order = if m.order == rowMajor: colMajor else: rowMajor
  shallowCopy(result.data, m.data)

proc reshape*[A](m: Matrix[A], a, b: int): Matrix[A] =
  assert(m.M * m.N == a * b, "The dimensions do not match: M = " & $(m.M) & ", N = " & $(m.N) & ", A = " & $(a) & ", B = " & $(b))
  new result
  result.M = a
  result.N = b
  result.order = m.order
  shallowCopy(result.data, m.data)

proc asMatrix*[A](v: Vector[A], a, b: int, order = colMajor): Matrix[A] =
  assert(v.len == a * b, "The dimensions do not match: N = " & $(v.len) & ", A = " & $(a) & ", B = " & $(b))
  new result
  result.order = order
  shallowCopy(result.data, v)
  result.M = a
  result.N = b

proc asVector*[A](m: Matrix[A]): Vector[A] =
  shallowCopy(result, m.data)

# BLAS level 1 operations

proc `*=`*[A: SomeReal](v: var Vector[A], k: A) {. inline .} = scal(v.len, k, v.fp, 1)

proc `*`*[A: SomeReal](v: Vector[A], k: A): Vector[A] {. inline .} =
  let N = v.len
  result = newSeq[A](N)
  copy(N, v.fp, 1, result.fp, 1)
  scal(N, k, result.fp, 1)

proc `+=`*[A: SomeReal](v: var Vector[A], w: Vector[A]) {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  axpy(N, 1, w.fp, 1, v.fp, 1)

proc `+`*[A: SomeReal](v, w: Vector[A]): Vector[A]  {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  result = newSeq[A](N)
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, 1, w.fp, 1, result.fp, 1)

proc `-=`*[A: SomeReal](v: var Vector[A], w: Vector[A]) {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  axpy(N, -1, w.fp, 1, v.fp, 1)

proc `-`*[A: SomeReal](v, w: Vector[A]): Vector[A]  {. inline .} =
  assert(v.len == w.len)
  let N = v.len
  result = newSeq[A](N)
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, -1, w.fp, 1, result.fp, 1)

proc `*`*[A: SomeReal](v, w: Vector[A]): A {. inline .} =
  assert(v.len == w.len)
  return dot(v.len, v.fp, 1, w.fp, 1)

proc l_2*[A: SomeReal](v: Vector[A]): auto {. inline .} = nrm2(v.len, v.fp, 1)

proc l_1*[A: SomeReal](v: Vector[A]): auto {. inline .} = asum(v.len, v.fp, 1)

proc maxIndex*[A](v: Vector[A]): tuple[i: int, val: A] =
  var
    j = 0
    m = v[0]
  for i, val in v:
    if val > m:
      j = i
      m = val
  return (j, m)

template max*[A](v: Vector[A]): A = maxIndex(v).val

proc minIndex*[A](v: Vector[A]): tuple[i: int, val: A] =
  var
    j = 0
    m = v[0]
  for i, val in v:
    if val < m:
      j = i
      m = val
  return (j, m)

template min*[A](v: Vector[A]): A = minIndex(v).val

template len(m: Matrix): int = m.M * m.N

template initLike[A](r, m: Matrix[A]) =
  new r
  r.data = newSeq[A](m.len)
  r.order = m.order
  r.M = m.M
  r.N = m.N

proc `*=`*[A: SomeReal](m: var Matrix[A], k: A) {. inline .} = scal(m.M * m.N, k, m.fp, 1)

proc `*`*[A: SomeReal](m: Matrix[A], k: A): Matrix[A]  {. inline .} =
  result.initLike(m)
  copy(m.len, m.fp, 1, result.fp, 1)
  scal(m.len, k, result.fp, 1)

template `*`*[A: SomeReal](k: A, v: Vector[A] or Matrix[A]): auto = v * k

template `/`*[A: SomeReal](k: A, v: Vector[A] or Matrix[A]): auto = v * (1 / k)

template `/=`*[A: SomeReal](v: var Vector[A] or var Matrix[A], k: A) =
  v *= (1 / k)

proc `+=`*[A: SomeReal](a: var Matrix[A], b: Matrix[A]) {. inline .} =
  assert a.M == b.M and a.N == a.N
  if a.order == b.order:
    axpy(a.M * a.N, 1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    for i in 0 ..< a.M:
      for j in 0 ..< a.N:
        a.data[j * a.M + i] += b.data[i * b.N + j]
  else:
    for i in 0 ..< a.M:
      for j in 0 ..< a.N:
        a.data[i * a.N + j] += b.data[j * b.M + i]

proc `+`*[A: SomeReal](a, b: Matrix[A]): Matrix[A] {. inline .} =
  result.initLike(a)
  copy(a.len, a.fp, 1, result.fp, 1)
  result += b

proc `-=`*[A: SomeReal](a: var Matrix[A], b: Matrix[A]) {. inline .} =
  assert a.M == b.M and a.N == a.N
  if a.order == b.order:
    axpy(a.M * a.N, -1, b.fp, 1, a.fp, 1)
  elif a.order == colMajor and b.order == rowMajor:
    for i in 0 ..< a.M:
      for j in 0 ..< a.N:
        a.data[j * a.M + i] -= b.data[i * b.N + j]
  else:
    for i in 0 ..< a.M:
      for j in 0 ..< a.N:
        a.data[i * a.N + j] -= b.data[j * b.M + i]

proc `-`*[A: SomeReal](a, b: Matrix[A]): Matrix[A] {. inline .} =
  result.initLike(a)
  copy(a.len, a.fp, 1, result.fp, 1)
  result -= b

proc l_2*[A: SomeReal](m: Matrix[A]): A {. inline .} = nrm2(m.len, m.fp, 1)

proc l_1*[A: SomeReal](m: Matrix[A]): A {. inline .} = asum(m.len, m.fp, 1)

template max*[A](m: Matrix[A]): A = max(m.data)

template min*[A](m: Matrix[A]): A = min(m.data)

# BLAS level 2 operations

proc `*`*[A: SomeReal](a: Matrix[A], v: Vector[A]): Vector[A]  {. inline .} =
  assert(a.N == v.len)
  result = newSeq[A](a.M)
  let lda = if a.order == colMajor: a.M.int else: a.N.int
  gemv(a.order, noTranspose, a.M, a.N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

# BLAS level 3 operations

proc `*`*[A: SomeReal](a, b: Matrix[A]): Matrix[A] {. inline .} =
  new result
  let
    M = a.M
    K = a.N
    N = b.N
  assert b.M == K
  result.data = newSeq[A](M * N)
  result.M = M
  result.N = N
  if a.order == colMajor and b.order == colMajor:
    result.order = colMajor
    gemm(colMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, M, b.fp, K, 0, result.fp, M)
  elif a.order == rowMajor and b.order == rowMajor:
    result.order = rowMajor
    gemm(rowMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, K, b.fp, N, 0, result.fp, N)
  elif a.order == colMajor and b.order == rowMajor:
    result.order = colMajor
    gemm(colMajor, noTranspose, transpose, M, N, K, 1, a.fp, M, b.fp, N, 0, result.fp, M)
  else:
    result.order = colMajor
    gemm(colMajor, transpose, noTranspose, M, N, K, 1, a.fp, K, b.fp, K, 0, result.fp, M)

# Comparison

template compareApprox(a, b: Vector or Matrix): bool =
  const epsilon = 0.000001
  let
    aNorm = l_1(a)
    bNorm = l_1(b)
    dNorm = l_1(a - b)
  (dNorm / (aNorm + bNorm)) < epsilon

proc `=~`*[A: SomeReal](v, w: Vector[A]): bool =
  compareApprox(v, w)

proc `=~`*[A: SomeReal](v, w: Matrix[A]): bool =
  compareApprox(v, w)

template `!=~`*(a, b: Vector or Matrix): bool =
  not (a =~ b)

# Hadamard (component-wise) product
proc `|*|`*[A](a, b: Vector[A]): Vector[A] =
  assert a.len == b.len
  result = newSeq[A](a.len)
  for i in 0 ..< N:
    result[i] = a[i] * b[i]

proc `|*|`*[A](a, b: Matrix[A]): Matrix[A] =
  assert a.dim == b.dim
  let (m, n) = a.dim
  result.initLike(a)
  if a.order == b.order:
    result.order = a.order
    for i in 0 ..< a.len:
      result.data[i] = a.data[i] * b.data[i]
  else:
    for i in 0 ..< a.M:
      for j in 0 ..< a.N:
        result[i, j] = a[i, j] * b[i, j]

# Universal functions

template makeUniversal*(fname: untyped) =
  when not compiles(fname(0'f32)):
    proc fname*(x: float32): float32 = fname(x.float64).float32

  proc fname*[A: SomeReal](v: Vector[A]): Vector[A] =
    result = newSeq[A](v.len)
    for i in 0 ..< (v.len):
      result[i] = fname(v[i])

  proc fname*[A: SomeReal](m: Matrix[A]): Matrix[A] =
    result.initLike(m)
    for i in 0 ..< m.len:
      result.data[i] = fname(m.data[i])

  export fname


template makeUniversalLocal*(fname: untyped) =
  when not compiles(fname(0'f32)):
    proc fname(x: float32): float32 = fname(x.float64).float32

  proc fname[A: SomeReal](v: Vector[A]): Vector[A] =
    result = newSeq[A](v.len)
    for i in 0 ..< (v.len):
      result[i] = fname(v[i])

  proc fname[A: SomeReal](m: Matrix[A]): Matrix[A] =
    result.initLike(m)
    for i in 0 ..< m.len:
      result.data[i] = fname(m.data[i])

makeUniversal(sqrt)
makeUniversal(cbrt)
makeUniversal(log10)
makeUniversal(log2)
makeUniversal(log)
makeUniversal(exp)
makeUniversal(arccos)
makeUniversal(arcsin)
makeUniversal(arctan)
makeUniversal(cos)
makeUniversal(cosh)
makeUniversal(sin)
makeUniversal(sinh)
makeUniversal(tan)
makeUniversal(tanh)
makeUniversal(erf)
makeUniversal(erfc)
makeUniversal(lgamma)
makeUniversal(tgamma)
makeUniversal(trunc)
makeUniversal(floor)
makeUniversal(ceil)
makeUniversal(degToRad)
makeUniversal(radToDeg)

# Functional API

proc cumsum*[A](v: Vector[A]): Vector[A] =
  result = newSeq[A](v.len)
  result[0] = v[0]
  for i in 1 ..< v.len:
    result[i] = result[i - 1] + v[i]

proc sum*[A](v: Vector[A]): A =
  foldl(v, a + b)

proc mean*[A: SomeReal](v: Vector[A]): A {.inline.} =
  sum(v) / A(v.len)

proc variance*[A: SomeReal](v: Vector[A]): A =
  let m = v.mean
  result = v[0] - v[0]
  for x in v:
    let y = x - m
    result += y * y
  result /= A(v.len)

template stddev*[A: SomeReal](v: Vector[A]): A =
  sqrt(variance(v))

template sum*(m: Matrix): auto = m.asVector.sum

template mean*(m: Matrix): auto = m.asVector.mean

template variance*(m: Matrix): auto = m.asVector.variance

template stddev*(m: Matrix): auto = m.asVector.stddev

# Rewrites

proc linearCombination[A: SomeReal](a: A, v, w: Vector[A]): Vector[A]  {. inline .} =
  new result
  copy(v.len, v.fp, 1, result.fp, 1)
  axpy(v.len, a, w.fp, 1, result.fp, 1)

proc linearCombinationMut[A: SomeReal](a: A, v: var Vector[A], w: Vector[A])  {. inline .} =
  axpy(v.len, a, w.fp, 1, v.fp, 1)

template rewriteLinearCombination*{v + `*`(w, a)}[A: SomeReal](a: A, v, w: Vector[A]): auto =
  linearCombination(a, v, w)

template rewriteLinearCombinationMut*{v += `*`(w, a)}[A: SomeReal](a: A, v: var Vector[A], w: Vector[A]): auto =
  linearCombinationMut(a, v, w)