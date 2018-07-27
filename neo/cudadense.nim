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

import nimcuda/[cuda_runtime_api, driver_types, cublas_api, cublas_v2, nimcuda]
import ./core, ./dense, ./private/neocommon

type
  CudaVector*[A] = object
    data*: ref[ptr A]
    fp*: ptr A
    len, step*: int32
  CudaMatrix*[A] = object
    M*, N*, ld*: int32
    data*: ref[ptr A]
    fp*: ptr A
    shape*: set[MatrixShape]

proc cudaMalloc[A](size: int): ptr A =
  let s = size * sizeof(A)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc freeDeviceMemory[A](p: ref[ptr A]) =
  if not p[].isNil:
    check cudaFree(p[])

proc isContiguous*(v: CudaVector): bool {.inline.} =
  v.step == 1

proc isContiguous*(m: CudaMatrix): bool {.inline.} =
  m.M == m.ld

# Initializing matrices

template init*[A](v: CudaVector[A], n: int) =
  new v.data, freeDeviceMemory
  v.fp = cudaMalloc[A](n)
  v.data[] = v.fp
  v.len = n.int32
  v.step = 1

template init*[A](v: CudaMatrix[A], m, n: int) =
  new v.data, freeDeviceMemory
  v.fp = cudaMalloc[A](m * n)
  v.data[] = v.fp
  v.M = m.int32
  v.N = n.int32
  v.ld = m.int32

proc newCudaVector*[A](n: int): CudaVector[A] {.inline.} =
  init(result, n)

proc newCudaMatrix*[A](m, n: int): CudaMatrix[A] {.inline.} =
  init(result, m, n)

# Copying between host and device

proc gpu*[A](v: Vector[A]): CudaVector[A] =
  init(result, v.len)
  check cublasSetVector(v.len.int32, sizeof(A).int32, v.fp, v.step.int32, result.fp, result.step)

proc gpu*[A](m: Matrix[A]): CudaMatrix[A] =
  if m.order == rowMajor:
    raise newException(ValueError, "m must be column major")
  init(result, m.M, m.N)
  check cublasSetMatrix(m.M.int32, m.N.int32, sizeof(A).int32, m.fp, m.ld.int32, result.fp, result.ld)

proc cpu*[A](v: CudaVector[A]): Vector[A] =
  result = zeros(v.len, A)
  check cublasGetVector(v.len, sizeof(A).int32, v.fp, v.step, result.fp, result.step.int32)

proc cpu*[A](m: CudaMatrix[A]): Matrix[A] =
  result = zeros(m.M, m.N, A, colMajor)
  check cublasGetMatrix(m.M, m.N, sizeof(A).int32, m.fp, m.ld, result.fp, result.ld.int32)

# Printing

proc `$`*[A](v: CudaVector[A]): string = $(v.cpu())

proc `$`*[A](m: CudaMatrix[A]): string = $(m.cpu())

# Equality

proc `==`*[A](m, n: CudaVector[A]): bool =
  m.cpu() == n.cpu()

proc `==`*[A](m, n: CudaMatrix[A]): bool =
  m.cpu() == n.cpu()

# Conversion

proc clone*[A](v: CudaVector[A]): CudaVector[A] =
  init(result, v.len)
  check cublasSetVector(v.len, sizeof(A).int32, v.fp, v.step, result.fp, result.step)

proc clone*[A](m: CudaMatrix[A]): CudaMatrix[A] =
  init(result, m.M, m.N)
  check cublasSetMatrix(m.M, m.N, sizeof(A).int32, m.fp, m.ld, result.fp, result.ld)

# CUBLAS overloads

var defaultHandle: cublasHandle_t
check cublasCreate_v2(addr defaultHandle)

overload(scal, cublasSscal, cublasDscal)
overload(axpy, cublasSaxpy, cublasDaxpy)
overload(asum, cublasSasum, cublasDasum)
overload(nrm2, cublasSnrm2, cublasDnrm2)
overload(dot, cublasSdot, cublasDdot)
overload(copy, cublasScopy, cublasDcopy)
overload(gemv, cublasSgemv, cublasDgemv)
overload(gemm, cublasSgemm, cublasDgemm)
overload(geam, cublasSgeam, cublasDgeam)

  # Slicing

proc `[]`*[A](v: CudaVector[A], s: Slice[int]): CudaVector[A] =
  checkBounds(s.a >= 0 and s.b < v.len)
  let
    vp = cast[CPointer[A]](v.fp)
    fp = addr(vp[s.a * v.step])
  result = CudaVector[A](
    len: (s.b - s.a + 1).int32,
    step: v.step,
    data: v.data,
    fp: fp
  )

proc pointerAt[A](v: CudaVector[A], i: int): ptr A {. inline .} =
  let s = cast[CPointer[A]](v.fp)
  addr s[i]

proc `[]=`*[A: SomeFloat](v: var CudaVector[A], s: Slice[int], val: CudaVector[A]) {. inline .} =
  checkBounds(s.a >= 0 and s.b < v.len)
  checkDim(s.len == val.len)
  check copy(defaultHandle, val.len, val.fp, val.step, v.pointerAt(s.a), v.step)

proc `[]`*[A](m: CudaMatrix[A], rows, cols: Slice[int]): CudaMatrix[A] =
  checkBounds(rows.a >= 0 and rows.b < m.M)
  checkBounds(cols.a >= 0 and cols.b < m.N)
  let
    mp = cast[CPointer[A]](m.fp)
    fp = addr(mp[cols.a * m.ld + rows.a])
  result = CudaMatrix[A](
    M: (rows.b - rows.a + 1).int32,
    N: (cols.b - cols.a + 1).int32,
    ld: m.ld,
    data: m.data,
    fp: fp
  )

proc `[]`*[A](m: CudaMatrix[A], rows: Slice[int], cols: typedesc[All]): CudaMatrix[A] =
  m[rows, 0 ..< m.N.int]

proc `[]`*[A](m: CudaMatrix[A], rows: typedesc[All], cols: Slice[int]): CudaMatrix[A] =
  m[0 ..< m.M.int, cols]

proc `[]=`*[A: SomeFloat](m: var CudaMatrix[A], rows, cols: Slice[int], val: CudaMatrix[A]) {. inline .} =
  checkBounds(rows.a >= 0 and rows.b < m.M)
  checkBounds(cols.a >= 0 and cols.b < m.N)
  checkDim(rows.len == val.M)
  checkDim(cols.len == val.N)
  let
    mp = cast[CPointer[A]](m.fp)
    vp = cast[CPointer[A]](val.fp)
  var col = 0
  for c in cols:
    check copy(defaultHandle, val.M, addr vp[col * val.ld], 1, addr mp[c * m.ld + rows.a], 1)
    col += 1

proc column*[A](m: CudaMatrix[A], j: int): CudaVector[A] {. inline .} =
  checkBounds(j >= 0 and j < m.N)
  let mp = cast[CPointer[A]](m.fp)
  result = CudaVector[A](
    data: m.data,
    fp: addr(mp[j * m.ld]),
    len: m.M,
    step: 1
  )

proc row*[A](m: CudaMatrix[A], i: int): CudaVector[A] {. inline .} =
  checkBounds(i >= 0 and i < m.M)
  let mp = cast[CPointer[A]](m.fp)
  result = CudaVector[A](
    data: m.data,
    fp: addr(mp[i]),
    len: m.N,
    step: m.ld
  )

# Iterators

iterator columns*[A](m: CudaMatrix[A]): auto {. inline .} =
  let mp = cast[CPointer[A]](m.fp)
  var v = m.column(0)
  yield v
  for j in 1 ..< m.N:
    v.fp = addr(mp[j * m.ld])
    yield v

iterator rows*[A](m: CudaMatrix[A]): auto {. inline .} =
  let mp = cast[CPointer[A]](m.fp)
  var v = m.row(0)
  yield v
  for i in 1 ..< m.M:
    v.fp = addr(mp[i])
    yield v

iterator columnsSlow*[A](m: CudaMatrix[A]): auto {. inline .} =
  for i in 0 ..< m.N:
    yield m.column(i)

iterator rowsSlow*[A](m: CudaMatrix[A]): auto {. inline .} =
  for i in 0 ..< m.M:
    yield m.row(i)

# Trivial operations

proc reshape*[A](m: CudaMatrix[A], a, b: int): CudaMatrix[A] =
  if m.isContiguous:
    checkDim(m.M * m.N == a * b, "The dimensions do not match: M = " & $(m.M) & ", N = " & $(m.N) & ", A = " & $(a) & ", B = " & $(b))
    result = CudaMatrix[A](
      M: a.int32,
      N: b.int32,
      ld: a.int32,
      data: m.data,
      fp: m.fp
    )
  else:
    result = m.clone().reshape(a, b)

proc asMatrix*[A](v: CudaVector[A], a, b: int): CudaMatrix[A] =
  if v.isContiguous:
    checkDim(v.len == a * b, "The dimensions do not match: N = " & $(v.len) & ", A = " & $(a) & ", B = " & $(b))
    result = CudaMatrix[A](
      M: a.int32,
      N: b.int32,
      ld: a.int32,
      data: v.data,
      fp: v.fp
    )
  else:
    result = v.clone().asMatrix(a, b)

proc asVector*[A](m: CudaMatrix[A]): CudaVector[A] =
  if m.isContiguous:
    CudaVector[A](
      len: m.M * m.N,
      step: 1,
      data: m.data,
      fp: m.fp
    )
  else:
    m.clone().asVector()

# BLAS level 1 operations

proc `*=`*[A: SomeFloat](v: var CudaVector[A], k: A) {. inline .} =
  var k1 = k
  check scal(defaultHandle, v.len, addr(k1), v.fp, v.step)

proc `*`*[A: SomeFloat](v: CudaVector[A], k: A): CudaVector[A]  {. inline .} =
  init(result, v.len)
  check copy(defaultHandle, v.len, v.fp, v.step, result.fp, result.step)
  result *= k

proc `+=`*[A: SomeFloat](v: var CudaVector[A], w: CudaVector[A]) {. inline .} =
  checkDim(v.len == w.len)
  var alpha: A = 1
  check axpy(defaultHandle, v.len, addr(alpha), w.fp, w.step, v.fp, v.step)

proc `+`*[A: SomeFloat](v, w: CudaVector[A]): CudaVector[A] {. inline .} =
  checkDim(v.len == w.len)
  init(result, v.len)
  check copy(defaultHandle, v.len, v.fp, v.step, result.fp, result.step)
  result += w

proc `-=`*[A: SomeFloat](v: var CudaVector[A], w: CudaVector[A]) {. inline .} =
  checkDim(v.len == w.len)
  var alpha: A = -1
  check axpy(defaultHandle, v.len, addr(alpha), w.fp, w.step, v.fp, v.step)

proc `-`*[A: SomeFloat](v, w: CudaVector[A]): CudaVector[A] {. inline .} =
  checkDim(v.len == w.len)
  init(result, v.len)
  check copy(defaultHandle, v.len, v.fp, v.step, result.fp, result.step)
  result -= w

proc `*`*[A: SomeFloat](v, w: CudaVector[A]): A {. inline .} =
  checkDim(v.len == w.len)
  check dot(defaultHandle, v.len, v.fp, v.step, w.fp, w.step, addr(result))

proc l_2*[A: SomeFloat](v: CudaVector[A]): A {. inline .} =
  check nrm2(defaultHandle, v.len, v.fp, v.step, addr(result))

proc l_1*[A: SomeFloat](v: CudaVector[A]): A {. inline .} =
  check asum(defaultHandle, v.len, v.fp, v.step, addr(result))

proc `*=`*[A: SomeFloat](m: var CudaMatrix[A], k: A) {. inline .} =
  var k1 = k
  if m.isContiguous:
    check scal(defaultHandle, m.M * m.N, addr(k1), m.fp, 1)
  else:
    for c in m.columns:
      check scal(defaultHandle, c.len, addr(k1), c.fp, c.step)

proc `*`*[A: SomeFloat](m: CudaMatrix[A], k: A): CudaMatrix[A]  {. inline .} =
  if m.isContiguous:
    init(result, m.M, m.N)
    check copy(defaultHandle, m.M * m.N, m.fp, 1, result.fp, 1)
  else:
    result = m.clone()
  result *= k

template `*`*[A: SomeFloat](k: A, v: CudaVector[A] or CudaMatrix[A]): auto =
  v * k

template `/`*[A: SomeFloat](v: CudaVector[A] or CudaMatrix[A], k: A): auto =
  v * (1 / k)

template `/=`*[A: SomeFloat](v: var CudaVector[A] or var CudaMatrix[A], k: A) =
  v *= (1 / k)

proc `+=`*[A: SomeFloat](a: var CudaMatrix[A], b: CudaMatrix[A]) {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  var alpha: A = 1
  check geam(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_N, a.M, a.N, addr(alpha),
    a.fp, a.ld, addr(alpha), b.fp, b.ld, a.fp, a.ld)

proc `+`*[A: SomeFloat](a, b: CudaMatrix[A]): CudaMatrix[A]  {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  init(result, a.M, a.N)
  var alpha: A = 1
  check geam(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_N, a.M, a.N, addr(alpha),
    a.fp, a.ld, addr(alpha), b.fp, b.ld, result.fp, result.ld)

proc `-=`*[A: SomeFloat](a: var CudaMatrix[A], b: CudaMatrix[A]) {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  var
    alpha: A = 1
    beta: A = -1
  check geam(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_N, a.M, a.N, addr(alpha),
    a.fp, a.ld, addr(beta), b.fp, b.ld, a.fp, a.ld)

proc `-`*[A: SomeFloat](a, b: CudaMatrix[A]): CudaMatrix[A]  {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  init(result, a.M, a.N)
  var
    alpha: A = 1
    beta: A = -1
  check geam(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_N, a.M, a.N, addr(alpha),
    a.fp, a.ld, addr(beta), b.fp, b.ld, result.fp, result.ld)

proc l_2*[A: SomeFloat](m: CudaMatrix[A]): A {. inline .} =
  if m.isContiguous:
    check nrm2(defaultHandle, m.M * m.N, m.fp, 1, addr(result))
  else:
    result = l_2(m.clone())

proc l_1*[A: SomeFloat](m: CudaMatrix[A]): A {. inline .} =
  if m.isContiguous:
    check asum(defaultHandle, m.M * m.N, m.fp, 1, addr(result))
  else:
    result = l_1(m.clone())

proc T*[A](m: CudaMatrix[A]): CudaMatrix[A] =
  init(result, m.N, m.M)
  var
    alpha: A = 1
    beta: A = 0
  check geam(defaultHandle, CUBLAS_OP_T, CUBLAS_OP_T, m.N, m.M, addr(alpha),
    m.fp, m.ld, addr(beta), m.fp, m.ld, result.fp, result.ld)

# BLAS level 2 operations

proc `*`*[A: SomeFloat](a: CudaMatrix[A], v: CudaVector[A]): CudaVector[A]  {. inline .} =
  checkDim(a.N == v.len)
  init(result, a.M)
  var
    alpha: A = 1
    beta: A = 0
  check gemv(defaultHandle, CUBLAS_OP_N, a.M, a.N, addr(alpha), a.fp, a.ld,
    v.fp, v.step, addr(beta), result.fp, result.step)

# BLAS level 3 operations

proc `*`*[A: SomeFloat](a, b: CudaMatrix[A]): CudaMatrix[A] {. inline .} =
  checkDim(a.N == b.M)
  init(result, a.M, b.N)
  var
    alpha: A = 1
    beta: A = 0
  let x = gemm(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_N, a.M, b.N, a.N,
    addr(alpha), a.fp, a.ld, b.fp, b.ld, addr(beta), result.fp, result.ld)

# Comparison

template compareApprox(a, b: CudaVector or CudaMatrix): bool =
  const epsilon = 0.000001
  let
    aNorm = l_1(a)
    bNorm = l_1(b)
    dNorm = l_1(a - b)
  (dNorm / (aNorm + bNorm)) < epsilon

proc `=~`*[A: SomeFloat](v, w: CudaVector[A]): bool = compareApprox(v, w)

proc `=~`*[A: SomeFloat](v, w: CudaMatrix[A]): bool = compareApprox(v, w)

template `!=~`*(a, b: CudaVector or CudaMatrix): bool = not (a =~ b)