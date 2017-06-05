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
import nimblas, nimlapack, sequtils, random, math
import ./core, ./private/neocommon

export nimblas.OrderType

type
  MatrixShape* = enum
    Diagonal, UpperTriangular, LowerTriangular, UpperHessenberg, LowerHessenberg, Symmetric
  Vector*[A] = ref object
    data*: seq[A]
    fp*: ptr A # float pointer
    len*, step*: int
  Matrix*[A] = ref object
    order*: OrderType
    M*, N*, ld*: int # ld = leading dimension
    fp*: ptr A # float pointer
    data*: seq[A]
    shape*: set[MatrixShape]

# Equality

proc isFull*(v: Vector): bool {.inline.} =
  v.len == v.data.len

proc isFull*(m: Matrix): bool {.inline.} =
  m.data.len == m.M * m.N

proc slowEq[A](v, w: Vector[A]): bool =
  if v.len != w.len:
    return false
  let
    vp = cast[CPointer[A]](v.fp)
    wp = cast[CPointer[A]](w.fp)
  for i in 0 ..< v.len:
    if vp[i * v.step] != wp[i * w.step]:
      return false
  return true

proc slowEq[A](m, n: Matrix[A]): bool =
  if m.M != n.M or m.N != n.N:
    return false
  let
    mp = cast[CPointer[A]](m.fp)
    np = cast[CPointer[A]](n.fp)
  for i in 0 ..< m.M:
    for j in 0 ..< m.N:
      let
        mel = if m.order == colMajor: mp[j * m.ld + i] else: mp[i * m.ld + j]
        nel = if n.order == colMajor: np[j * n.ld + i] else: np[i * n.ld + j]
      if mel != nel:
        return false
  return true

proc `==`*[A](v, w: Vector[A]): bool =
  if v.isFull and w.isFull:
    v.data == w.data
  else:
    slowEq(v, w)

proc `==`*[A](m, n: Matrix[A]): bool =
  if m.isFull and n.isFull and m.order == n.order:
    m.data == n.data
  else:
    slowEq(m, n)

# Initializers

proc vector*[A](data: seq[A]): Vector[A] =
  result = Vector[A](step: 1, len: data.len)
  shallowCopy(result.data, data)
  result.fp = addr(result.data[0])

proc vector*[A](data: varargs[A]): Vector[A] =
  vector[A](@data)

proc makeVector*[A](N: int, f: proc (i: int): A): Vector[A] =
  result = vector(newSeq[A](N))
  for i in 0 ..< N:
    result.data[i] = f(i)

template makeVectorI*[A](N: int, f: untyped): Vector[A] =
  var result = vector(newSeq[A](N))
  for i {.inject.} in 0 ..< N:
    result.data[i] = f
  result

proc randomVector*(N: int, max: float64 = 1): Vector[float64] =
  makeVectorI[float64](N, random(max))

proc randomVector*(N: int, max: float32): Vector[float32] =
  makeVectorI[float32](N, random(max).float32)

proc constantVector*[A](N: int, a: A): Vector[A] = makeVectorI[A](N, a)

proc zeros*(N: int): auto = vector(newSeq[float64](N))

proc zeros*(N: int, A: typedesc): auto = vector(newSeq[A](N))

proc ones*(N: int): auto = constantVector(N, 1'f64)

proc ones*(N: int, A: typedesc[float32]): auto = constantVector(N, 1'f32)

proc ones*(N: int, A: typedesc[float64]): auto = constantVector(N, 1'f64)

proc matrix*[A](order: OrderType, M, N: int, data: seq[A]): Matrix[A] =
  result = Matrix[A](
    order: order,
    M: M,
    N: N,
    ld: if order == rowMajor: N else: M
  )
  shallowCopy(result.data, data)
  result.fp = addr(result.data[0])

proc makeMatrix*[A](M, N: int, f: proc (i, j: int): A, order = colMajor): Matrix[A] =
  result = matrix[A](order, M, N, newSeq[A](M * N))
  if order == colMajor:
    for i in 0 ..< M:
      for j in 0 ..< N:
        result.data[j * M + i] = f(i, j)
  else:
    for i in 0 ..< M:
      for j in 0 ..< N:
        result.data[i * N + j] = f(i, j)

template makeMatrixIJ*(A: typedesc, M1, N1: int, f: untyped, ord = colMajor): auto =
  var r = matrix[A](ord, M1, N1, newSeq[A](M1 * N1))
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
  result = matrix[A](order, M, N, newSeq[A](M * N))
  for i in 0 ..< (M * N):
    result.data[i] = random(max)

proc randomMatrix*(M, N: int, order = colMajor): Matrix[float64] =
  randomMatrix(M, N, 1'f64, order)

proc constantMatrix*[A](M, N: int, a: A, order = colMajor): Matrix[A] =
  matrix[A](order, M, N, sequtils.repeat(a, M * N))

proc zeros*(M, N: int, order = colMajor): auto =
  matrix[float64](M = M, N = N, order = order, data = newSeq[float64](M * N))

proc zeros*(M, N: int, A: typedesc, order = colMajor): auto =
  matrix[A](M = M, N = N, order = order, data = newSeq[A](M * N))

proc ones*(M, N: int): auto = constantMatrix(M, N, 1'f64)

proc ones*(M, N: int, A: typedesc[float32]): auto = constantMatrix(M, N, 1'f32)

proc ones*(M, N: int, A: typedesc[float64]): auto = constantMatrix(M, N, 1'f64)

proc eye*(N: int, order = colMajor): Matrix[float64] =
  makeMatrixIJ(float64, N, N, if i == j: 1 else: 0, order)

proc eye*(N: int, A: typedesc[float32], order = colMajor): Matrix[float32] =
  makeMatrixIJ(float32, N, N, if i == j: 1 else: 0, order)

proc eye*(N: int, A: typedesc[float64], order = colMajor): Matrix[float64] =
  makeMatrixIJ(float64, N, N, if i == j: 1 else: 0, order)

proc matrix*[A](xs: seq[seq[A]], order = colMajor): Matrix[A] =
  makeMatrixIJ(A, xs.len, xs[0].len, xs[i][j], order)

proc diag*[A: SomeReal](xs: varargs[A]): Matrix[A] =
  let n = xs.len
  result = zeros(n, n, A)
  for i in 0 ..< n:
    result.data[i * (n + 1)] = xs[i]

# Conversion

proc to32*(v: Vector[float64]): Vector[float32] =
  vector(v.data.mapIt(it.float32))

proc to64*(v: Vector[float32]): Vector[float64] =
  vector(v.data.mapIt(it.float64))

proc to32*(m: Matrix[float64]): Matrix[float32] =
  matrix(data = m.data.mapIt(it.float32), order = m.order, M = m.M, N = m.N)

proc to64*(m: Matrix[float32]): Matrix[float64] =
  matrix(data = m.data.mapIt(it.float64), order = m.order, M = m.M, N = m.N)

# Accessors

proc `[]`*[A](v: Vector[A], i: int): A {. inline .} =
  v.data[i]

proc `[]=`*[A](v: Vector[A], i: int, val: A) {. inline .} =
  v.data[i] = val

proc `[]`*[A](m: Matrix[A], i, j: int): A {. inline .} =
  if m.order == colMajor: m.data[j * m.M + i]
  else: m.data[i * m.N + j]

proc `[]=`*[A](m: var Matrix[A], i, j: int, val: A) {. inline .} =
  if m.order == colMajor:
    m.data[j * m.M + i] = val
  else:
    m.data[i * m.N + j] = val

proc column*[A](m: Matrix[A], j: int): Vector[A] {. inline .} =
  result = zeros(m.M, A)
  for i in 0 ..< m.M:
    result.data[i] = m[i, j]

proc row*[A](m: Matrix[A], i: int): Vector[A] {. inline .} =
  result = zeros(m.N, A)
  for j in 0 ..< m.N:
    result.data[j] = m[i, j]

proc dim*(m: Matrix): tuple[rows, columns: int] = (m.M, m.N)

proc clone*[A](v: Vector[A]): Vector[A] =
  var dataCopy = v.data
  return vector(dataCopy)

proc clone*[A](m: Matrix[A]): Matrix[A] =
  var dataCopy = m.data
  return matrix[A](data = dataCopy, order = m.order, M = m.M, N = m.N)

proc map*[A](v: Vector[A], f: proc(x: A): A): Vector[A] =
  result = zeros(v.len, A)
  for i in 0 ..< v.len:
    result.data[i] = f(v.data[i])

proc map*[A](m: Matrix[A], f: proc(x: A): A): Matrix[A] =
  result = zeros(m.M, m.N, A, m.order)
  for i in 0 ..< (m.M * m.N):
    result.data[i] = f(m.data[i])

# Iterators

iterator items*[A](v: Vector[A]): auto {. inline .} =
  for x in v.data:
    yield x

iterator pairs*[A](v: Vector[A]): auto {. inline .} =
  for i, x in v.data:
    yield (i, x)

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

proc `$`*[A](v: Vector[A]): string =
  result = "[ "
  for i in 0 ..< (v.len - 1):
    result &= $(v[i]) & "\t"
  result &= $(v[v.len - 1]) & " ]"

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
  matrix(
    order = (if m.order == rowMajor: colMajor else: rowMajor),
    M = m.N,
    N = m.M,
    data = m.data
  )

proc reshape*[A](m: Matrix[A], a, b: int): Matrix[A] =
  checkDim(m.M * m.N == a * b, "The dimensions do not match: M = " & $(m.M) & ", N = " & $(m.N) & ", A = " & $(a) & ", B = " & $(b))
  result = matrix(
    order = m.order,
    M = a,
    N = b,
    data = m.data
  )

proc asMatrix*[A](v: Vector[A], a, b: int, order = colMajor): Matrix[A] =
  checkDim(v.len == a * b, "The dimensions do not match: N = " & $(v.len) & ", A = " & $(a) & ", B = " & $(b))
  result = matrix(
    order = order,
    M = a,
    N = b,
    data = v.data
  )

proc asVector*[A](m: Matrix[A]): Vector[A] =
  vector(m.data)

# BLAS level 1 operations

proc `*=`*[A: SomeReal](v: var Vector[A], k: A) {. inline .} = scal(v.len, k, v.fp, 1)

proc `*`*[A: SomeReal](v: Vector[A], k: A): Vector[A] {. inline .} =
  let N = v.len
  result = vector(newSeq[A](N))
  copy(N, v.fp, 1, result.fp, 1)
  scal(N, k, result.fp, 1)

proc `+=`*[A: SomeReal](v: var Vector[A], w: Vector[A]) {. inline .} =
  checkDim(v.len == w.len)
  let N = v.len
  axpy(N, 1, w.fp, 1, v.fp, 1)

proc `+`*[A: SomeReal](v, w: Vector[A]): Vector[A]  {. inline .} =
  checkDim(v.len == w.len)
  let N = v.len
  result = vector(newSeq[A](N))
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, 1, w.fp, 1, result.fp, 1)

proc `-=`*[A: SomeReal](v: var Vector[A], w: Vector[A]) {. inline .} =
  checkDim(v.len == w.len)
  let N = v.len
  axpy(N, -1, w.fp, 1, v.fp, 1)

proc `-`*[A: SomeReal](v, w: Vector[A]): Vector[A]  {. inline .} =
  checkDim(v.len == w.len)
  let N = v.len
  result = vector(newSeq[A](N))
  copy(N, v.fp, 1, result.fp, 1)
  axpy(N, -1, w.fp, 1, result.fp, 1)

proc `*`*[A: SomeReal](v, w: Vector[A]): A {. inline .} =
  checkDim(v.len == w.len)
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
  r = matrix[A](m.order, m.M, m.N, newSeq[A](m.len))

proc `*=`*[A: SomeReal](m: var Matrix[A], k: A) {. inline .} = scal(m.M * m.N, k, m.fp, 1)

proc `*`*[A: SomeReal](m: Matrix[A], k: A): Matrix[A]  {. inline .} =
  result.initLike(m)
  copy(m.len, m.fp, 1, result.fp, 1)
  scal(m.len, k, result.fp, 1)

template `*`*[A: SomeReal](k: A, v: Vector[A] or Matrix[A]): auto = v * k

template `/`*[A: SomeReal](v: Vector[A] or Matrix[A], k: A): auto = v * (1 / k)

template `/=`*[A: SomeReal](v: var Vector[A] or var Matrix[A], k: A) =
  v *= (1 / k)

proc `+=`*[A: SomeReal](a: var Matrix[A], b: Matrix[A]) {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
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
  checkDim(a.M == b.M and a.N == a.N)
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
  checkDim(a.N == v.len)
  result = vector(newSeq[A](a.M))
  let lda = if a.order == colMajor: a.M.int else: a.N.int
  gemv(a.order, noTranspose, a.M, a.N, 1, a.fp, lda, v.fp, 1, 0, result.fp, 1)

# BLAS level 3 operations

proc `*`*[A: SomeReal](a, b: Matrix[A]): Matrix[A] {. inline .} =
  let
    M = a.M
    K = a.N
    N = b.N
  checkDim(b.M == K)
  result = matrix[A](a.order, M, N, newSeq[A](M * N))
  if a.order == colMajor and b.order == colMajor:
    gemm(colMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, M, b.fp, K, 0, result.fp, M)
  elif a.order == rowMajor and b.order == rowMajor:
    gemm(rowMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, K, b.fp, N, 0, result.fp, N)
  elif a.order == colMajor and b.order == rowMajor:
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
  checkDim(a.len == b.len)
  result = vector(newSeq[A](a.len))
  for i in 0 ..< a.len:
    result[i] = a[i] * b[i]

proc `|*|`*[A](a, b: Matrix[A]): Matrix[A] =
  checkDim(a.dim == b.dim)
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
    result = vector(newSeq[A](v.len))
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
    result = vector(newSeq[A](v.len))
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
  result = vector(newSeq[A](v.len))
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
  result = vector(newSeq[A](v.N))
  copy(v.len, v.fp, 1, result.fp, 1)
  axpy(v.len, a, w.fp, 1, result.fp, 1)

proc linearCombinationMut[A: SomeReal](a: A, v: var Vector[A], w: Vector[A])  {. inline .} =
  axpy(v.len, a, w.fp, 1, v.fp, 1)

template rewriteLinearCombination*{v + `*`(w, a)}[A: SomeReal](a: A, v, w: Vector[A]): auto =
  linearCombination(a, v, w)

template rewriteLinearCombinationMut*{v += `*`(w, a)}[A: SomeReal](a: A, v: var Vector[A], w: Vector[A]): auto =
  linearCombinationMut(a, v, w)

# LAPACK overloads

overload(gesv, sgesv, dgesv)
overload(gebal, sgebal, dgebal)
overload(gehrd, sgehrd, dgehrd)
overload(orghr, sorghr, dorghr)
overload(hseqr, shseqr, dhseqr)

# Solvers

template solveMatrix(M, N, a, b: untyped): auto =
  assert(a.order == colMajor, "`solve` requires a column-major matrix")
  assert(b.order == colMajor, "`solve` requires a column-major matrix")

  var
    ipvt = newSeq[int32](M)
    info: cint
    m = M.cint
    n = N.cint
  fortran(gesv, m, n, a, m, ipvt, b, m, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Left hand matrix is singular or factorization failed")

template solveVector(M, a, b: untyped): auto =
  assert(a.order == colMajor, "`solve` requires a column-major matrix")
  var
    ipvt = newSeq[int32](M)
    info: cint
    m = M.cint
    n = 1.cint
  fortran(gesv, m, n, a, m, ipvt, b, m, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Left hand matrix is singular or factorization failed")

proc solve*[A: SomeReal](a, b: Matrix[A]): Matrix[A] {.inline.} =
  checkDim(a.M == a.N, "Need a square matrix to solve the system")
  checkDim(a.M == b.M, "The dimensions are incompatible")
  result = zeros(b.M, b.N, A, b.order)
  var acopy = a.clone
  copy(b.M * b.N, b.fp, 1, result.fp, 1)
  solveMatrix(b.M, b.N, acopy, result)

proc solve*[A: SomeReal](a: Matrix[A], b: Vector[A]): Vector[A] {.inline.} =
  checkDim(a.M == a.N, "Need a square matrix to solve the system")
  result = zeros(a.M, A)
  var acopy = a.clone
  copy(a.M, b.fp, 1, result.fp, 1)
  solveVector(a.M, acopy, result)

template `\`*(a: Matrix, b: Matrix or Vector): auto = solve(a, b)

proc inv*[A: SomeReal](a: Matrix[A]): Matrix[A] {.inline.} =
  checkDim(a.M == a.N, "Need a square matrix to invert")
  result = eye(a.M, A)
  var acopy = a.clone
  solveMatrix(a.M, a.M, acopy, result)

# Eigenvalues

type
  BalanceOp* {.pure.} = enum
    NoOp, Permute, Scale, Both
  EigenMode* {.pure.} = enum
    Eigenvalues, Schur
  SchurCompute* {.pure.} = enum
    NoOp, Initialize, Provided
  BalanceResult*[A] = object
    matrix*: Matrix[A]
    ihi*, ilo*: cint
    scale*: seq[A]
  EigenValues*[A] = ref object
    real*, img*: seq[A]
  SchurResult*[A] = object
    factorization*: Matrix[A]
    eigenvalues*: EigenValues[A]

proc ch(op: BalanceOp): char =
  case op
  of BalanceOp.NoOp: 'N'
  of BalanceOp.Permute: 'P'
  of BalanceOp.Scale: 'S'
  of BalanceOp.Both: 'B'

proc ch(m: EigenMode): char =
  case m
  of EigenMode.Eigenvalues: 'E'
  of EigenMode.Schur: 'S'

proc ch(s: SchurCompute): char =
  case s
  of SchurCompute.NoOp: 'N'
  of SchurCompute.Initialize: 'I'
  of SchurCompute.Provided: 'V'

proc balance*[A: SomeReal](a: Matrix[A], op = BalanceOp.Both): BalanceResult[A] =
  checkDim(a.M == a.N, "`balance` requires a square matrix")
  assert(a.order == colMajor, "`balance` requires a column-major matrix")
  result = BalanceResult[A](
    matrix: a.clone(),
    scale: newSeq[A](a.N)
  )

  var
    job = ch(op)
    n = a.N.cint
    info: cint
  # gebal(addr job, addr n, result.matrix.fp, addr n, addr result.ilo, addr result.ihi, result.scale.first, addr info)
  fortran(gebal, job, n, result.matrix, n, result.ilo, result.ihi, result.scale, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to balance matrix")

proc hessenberg*[A: SomeReal](a: Matrix[A]): Matrix[A] =
  checkDim(a.M == a.N, "`hessenberg` requires a square matrix")
  assert(a.order == colMajor, "`hessenberg` requires a column-major matrix")
  result = a.clone()
  var
    n = a.N.cint
    ilo = 1.cint
    ihi = a.N.cint
    info: cint
    tau = newSeq[A](a.N)
    workSize = (-1).cint
    work = newSeq[A](1)
  # First, we call gehrd to compute the optimal work size
  # gehrd(addr n, addr ilo, addr ihi, result.fp, addr n, tau.first, work.first, addr workSize, addr info)
  fortran(gehrd, n, ilo, ihi, result, n, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to reduce matrix to upper Hessenberg form")
  # Then, we allocate suitable space and call gehrd again
  workSize = work[0].cint
  work = newSeq[A](workSize)
  fortran(gehrd, n, ilo, ihi, result, n, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to reduce matrix to upper Hessenberg form")

proc eigenvalues*[A: SomeReal](a: Matrix[A]): EigenValues[A] =
  checkDim(a.M == a.N, "`eigenvalues` requires a square matrix")
  assert(a.order == colMajor, "`eigenvalues` requires a column-major matrix")
  var
    h = a.clone()
    n = a.N.cint
    ilo = 1.cint
    ihi = a.N.cint
    info: cint
    tau = newSeq[A](a.N)
    workSize = (-1).cint
    work = newSeq[A](1)
  # First, we call gehrd to compute the optimal work size
  fortran(gehrd, n, ilo, ihi, h, n, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find eigenvalues")
  # Then, we allocate suitable space and call gehrd again
  workSize = work[0].cint
  work = newSeq[A](workSize)
  fortran(gehrd, n, ilo, ihi, h, n, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find eigenvalues")
  # Next, we need to find the matrix Q that transforms A into H
  var q = h.clone()
  fortran(orghr, n, ilo, ihi, q, n, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find eigenvalues")
  var
    job = ch(EigenMode.Eigenvalues)
    compz = ch(SchurCompute.Provided)
  result = EigenValues[A](
    real: newSeq[A](a.N),
    img: newSeq[A](a.N)
  )
  fortran(hseqr, job, compz, n, ilo, ihi, h, n, result.real, result.img, q, n, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find eigenvalues")

proc schur*[A: SomeReal](a: Matrix[A]): SchurResult[A] =
  checkDim(a.M == a.N, "`schur` requires a square matrix")
  assert(a.order == colMajor, "`schur` requires a column-major matrix")
  result.factorization = a.clone()
  var
    n = a.N.cint
    ilo = 1.cint
    ihi = a.N.cint
    info: cint
    tau = newSeq[A](a.N)
    workSize = (-1).cint
    work = newSeq[A](1)
  # First, we call gehrd to compute the optimal work size
  fortran(gehrd, n, ilo, ihi, result.factorization, n, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find the Schur decomposition")
  # Then, we allocate suitable space and call gehrd again
  workSize = work[0].cint
  work = newSeq[A](workSize)
  fortran(gehrd, n, ilo, ihi, result.factorization, n, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find the Schur decomposition")
  # Next, we need to find the matrix Q that transforms A into H
  var q = result.factorization.clone()
  fortran(orghr, n, ilo, ihi, q, n, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find the Schur decomposition")
  var
    job = ch(EigenMode.Eigenvalues)
    compz = ch(SchurCompute.Provided)
  result.eigenvalues = EigenValues[A](
    real: newSeq[A](a.N),
    img: newSeq[A](a.N)
  )
  fortran(hseqr, job, compz, n, ilo, ihi, result.factorization, n,
    result.eigenvalues.real, result.eigenvalues.img, q, n, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find the Schur decomposition")