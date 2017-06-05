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
import ./dense, ./private/neocommon

type
  CudaVector*[A] = object
    N*: int32
    data*: ref[ptr A]
  CudaMatrix*[A] = object
    M*, N*: int32
    data*: ref[ptr A]

template fp[A](c: CudaVector[A] or CudaMatrix[A]): ptr A = c.data[]

template fp[A](v: Vector[A]): ptr A = cast[ptr A](unsafeAddr(v[0]))

template fp[A](m: Matrix[A]): ptr A = cast[ptr A](unsafeAddr(m.data[0]))

proc cudaMalloc[A](size: int): ptr A =
  let s = size * sizeof(A)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc freeDeviceMemory[A: SomeReal](p: ref[ptr A]) =
  check cudaFree(p[])

# Initializing matrices

template init*[A](v: CudaVector[A], n: int) =
  new v.data, freeDeviceMemory
  v.data[] = cudaMalloc[A](n)
  v.N = n.int32

template init*[A](v: CudaMatrix[A], m, n: int) =
  new v.data, freeDeviceMemory
  v.data[] = cudaMalloc[A](m * n)
  v.M = m.int32
  v.N = n.int32

proc newCudaVector*[A](n: int): CudaVector[A] {.inline.} =
  init(result, n)

proc newCudaMatrix*[A](m, n: int): CudaMatrix[A] {.inline.} =
  init(result, m, n)

# Copying between host and device

proc gpu*[A: SomeReal](v: Vector[A]): CudaVector[A] =
  init(result, v.len)
  check cublasSetVector(v.len.int32, sizeof(A).int32, v.fp, 1, result.fp, 1)

proc gpu*[A: SomeReal](m: Matrix[A]): CudaMatrix[A] =
  if m.order == rowMajor:
    raise newException(ValueError, "m must be column major")
  init(result, m.M, m.N)
  check cublasSetMatrix(m.M.int32, m.N.int32, sizeof(A).int32, m.fp, m.M.int32, result.fp, m.M.int32)

proc cpu*[A: SomeReal](v: CudaVector[A]): Vector[A] =
  result = zeros(v.N, A)
  check cublasGetVector(v.N, sizeof(A).int32, v.fp, 1, result.fp, 1)

proc cpu*[A: SomeReal](m: CudaMatrix[A]): Matrix[A] =
  result = zeros(m.M, m.N, A, colMajor)
  check cublasGetMatrix(m.M, m.N, sizeof(A).int32, m.fp, m.M, result.fp, m.M)

# Printing

proc `$`*[A](v: CudaVector[A]): string = $(v.cpu())

proc `$`*[A](m: CudaMatrix[A]): string = $(m.cpu())

# Equality

proc `==`*[A](m, n: CudaVector[A]): bool =
  m.cpu() == n.cpu()

proc `==`*[A](m, n: CudaMatrix[A]): bool =
  m.cpu() == n.cpu()

# BLAS overloads

var defaultHandle: cublasHandle_t
check cublasCreate_v2(addr defaultHandle)

proc cublasScal(handle: cublasHandle_t, n: int32, alpha: float32, x: ptr float32): cublasStatus_t =
  cublasSscal(handle, n, unsafeAddr(alpha), x, 1)

proc cublasScal(handle: cublasHandle_t, n: int32, alpha: float64, x: ptr float64): cublasStatus_t =
  cublasDscal(handle, n, unsafeAddr(alpha), x, 1)

proc cublasAxpy(handle: cublasHandle_t, n: int32, alpha: float32, x, y: ptr float32): cublasStatus_t =
  cublasSaxpy(handle, n, unsafeAddr(alpha), x, 1, y, 1)

proc cublasAxpy(handle: cublasHandle_t, n: int32, alpha: float64, x, y: ptr float64): cublasStatus_t =
  cublasDaxpy(handle, n, unsafeAddr(alpha), x, 1, y, 1)

proc cublasAsum(handle: cublasHandle_t, n: int32, x: ptr float32, incx: int32, y: ptr float32): cublasStatus_t =
  cublasSasum(handle, n, x, incx, y)

proc cublasAsum(handle: cublasHandle_t, n: int32, x: ptr float64, incx: int32, y: ptr float64): cublasStatus_t =
  cublasDasum(handle, n, x, incx, y)

proc cublasNrm2(handle: cublasHandle_t, n: int32, x: ptr float32, incx: int32, y: ptr float32): cublasStatus_t =
  cublasSnrm2(handle, n, x, incx, y)

proc cublasNrm2(handle: cublasHandle_t, n: int32, x: ptr float64, incx: int32, y: ptr float64): cublasStatus_t =
  cublasDnrm2(handle, n, x, incx, y)

proc cublasDot(handle: cublasHandle_t, n: int32, x: ptr float32, incx: int32, y: ptr float32, incy: int32, res: ptr float32): cublasStatus_t =
  cublasSdot(handle, n, x, incx, y, incy, res)

proc cublasDot(handle: cublasHandle_t, n: int32, x: ptr float64, incx: int32, y: ptr float64, incy: int32, res: ptr float64): cublasStatus_t =
  cublasDdot(handle, n, x, incx, y, incy, res)

proc cublasCopy(handle: cublasHandle_t, n: int32, x: ptr float32, incx: int32, y: ptr float32, incy: int32): cublasStatus_t =
  cublasScopy(handle, n, x, incx, y, incy)

proc cublasCopy(handle: cublasHandle_t, n: int32, x: ptr float64, incx: int32, y: ptr float64, incy: int32): cublasStatus_t =
  cublasDcopy(handle, n, x, incx, y, incy)

proc cublasGemv(handle: cublasHandle_t, trans: cublasOperation_t,
  m, n: int32, alpha: float32, A: ptr float32, lda: int32, x: ptr float32, incx: int32,
  beta: float32, y: ptr float32, incy: int32): cublasStatus_t =
  cublasSgemv(handle, trans, m, n, unsafeAddr(alpha), A, lda, x, incx, unsafeAddr(beta), y, incy)

proc cublasGemv(handle: cublasHandle_t, trans: cublasOperation_t,
  m, n: int32, alpha: float64, A: ptr float64, lda: int32, x: ptr float64, incx: int32,
  beta: float64, y: ptr float64, incy: int32): cublasStatus_t =
  cublasDgemv(handle, trans, m, n, unsafeAddr(alpha), A, lda, x, incx, unsafeAddr(beta), y, incy)

proc cublasGemm(handle: cublasHandle_t, transa, transb: cublasOperation_t,
  m, n, k: int32, alpha: float32, A: ptr float32, lda: int32, B: ptr float32,
  ldb: int32, beta: float32, C: ptr float32, ldc: int32): cublasStatus_t =
  cublasSgemm(handle, transa, transb, m, n, k, unsafeAddr(alpha), A, lda, B, ldb, unsafeAddr(beta), C, ldc)

proc cublasGemm(handle: cublasHandle_t, transa, transb: cublasOperation_t,
  m, n, k: int32, alpha: float64, A: ptr float64, lda: int32, B: ptr float64,
  ldb: int32, beta: float64, C: ptr float64, ldc: int32): cublasStatus_t =
  cublasDgemm(handle, transa, transb, m, n, k, unsafeAddr(alpha), A, lda, B, ldb, unsafeAddr(beta), C, ldc)

# BLAS level 1 operations

proc `*=`*[A: SomeReal](v: var CudaVector[A], k: A) {. inline .} =
  check cublasScal(defaultHandle, v.N, k, v.fp)

proc `*`*[A: SomeReal](v: CudaVector[A], k: A): CudaVector[A]  {. inline .} =
  init(result, v.N)
  check cublasCopy(defaultHandle, v.N, v.fp, 1, result.fp, 1)
  check cublasScal(defaultHandle, v.N, k, result.fp)

proc `+=`*[A: SomeReal](v: var CudaVector[A], w: CudaVector[A]) {. inline .} =
  checkDim(v.N == w.N)
  check cublasAxpy(defaultHandle, v.N, 1, w.fp, v.fp)

proc `+`*[A: SomeReal](v, w: CudaVector[A]): CudaVector[A] {. inline .} =
  checkDim(v.N == w.N)
  init(result, v.N)
  check cublasCopy(defaultHandle, v.N, v.fp, 1, result.fp, 1)
  check cublasAxpy(defaultHandle, v.N, 1, w.fp, result.fp)

proc `-=`*[A: SomeReal](v: var CudaVector[A], w: CudaVector[A]) {. inline .} =
  checkDim(v.N == w.N)
  check cublasAxpy(defaultHandle, v.N, -1, w.fp, v.fp)

proc `-`*[A: SomeReal](v, w: CudaVector[A]): CudaVector[A] {. inline .} =
  checkDim(v.N == w.N)
  init(result, v.N)
  check cublasCopy(defaultHandle, v.N, v.fp, 1, result.fp, 1)
  check cublasAxpy(defaultHandle, v.N, -1, w.fp, result.fp)

proc `*`*[A: SomeReal](v, w: CudaVector[A]): A {. inline .} =
  checkDim(v.N == w.N)
  check cublasDot(defaultHandle, v.N, v.fp, 1, w.fp, 1, addr(result))

proc l_2*[A: SomeReal](v: CudaVector[A]): A {. inline .} =
  check cublasNrm2(defaultHandle, v.N, v.fp, 1, addr(result))

proc l_1*[A: SomeReal](v: CudaVector[A]): A {. inline .} =
  check cublasAsum(defaultHandle, v.N, v.fp, 1, addr(result))

proc `*=`*[A: SomeReal](m: var CudaMatrix[A], k: A) {. inline .} =
  check cublasScal(defaultHandle, m.M * m.N, k, m.fp)

proc `*`*[A: SomeReal](m: CudaMatrix[A], k: A): CudaMatrix[A]  {. inline .} =
  init(result, m.M, m.N)
  check cublasCopy(defaultHandle, m.M * m.N, m.fp, 1, result.fp, 1)
  check cublasScal(defaultHandle, m.M * m.N, k, result.fp)

template `*`*[A: SomeReal](k: A, v: CudaVector[A] or CudaMatrix[A]): auto =
  v * k

template `/`*[A: SomeReal](v: CudaVector[A] or CudaMatrix[A], k: A): auto =
  v * (1 / k)

template `/=`*[A: SomeReal](v: var CudaVector[A] or var CudaMatrix[A], k: A) =
  v *= (1 / k)

proc `+=`*[A: SomeReal](a: var CudaMatrix[A], b: CudaMatrix[A]) {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  check cublasAxpy(defaultHandle, a.M * a.N, 1, b.fp, a.fp)

proc `+`*[A: SomeReal](a, b: CudaMatrix[A]): CudaMatrix[A]  {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  init(result, a.M, a.N)
  check cublasCopy(defaultHandle, a.M * a.N, a.fp, 1, result.fp, 1)
  check cublasAxpy(defaultHandle, a.M * a.N, 1, b.fp, result.fp)

proc `-=`*[A: SomeReal](a: var CudaMatrix[A], b: CudaMatrix[A]) {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  check cublasAxpy(defaultHandle, a.M * a.N, -1, b.fp, a.fp)

proc `-`*[A: SomeReal](a, b: CudaMatrix[A]): CudaMatrix[A]  {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  init(result, a.M, a.N)
  check cublasCopy(defaultHandle, a.M * a.N, a.fp, 1, result.fp, 1)
  check cublasAxpy(defaultHandle, a.M * a.N, -1, b.fp, result.fp)

proc l_2*[A: SomeReal](m: CudaMatrix[A]): A {. inline .} =
  check cublasNrm2(defaultHandle, m.M * m.N, m.fp, 1, addr(result))

proc l_1*[A: SomeReal](m: CudaMatrix[A]): A {. inline .} =
  check cublasAsum(defaultHandle, m.M * m.N, m.fp, 1, addr(result))

# BLAS level 2 operations

proc `*`*[A: SomeReal](a: CudaMatrix[A], v: CudaVector[A]): CudaVector[A]  {. inline .} =
  checkDim(a.N == v.N)
  init(result, a.M)
  check cublasGemv(defaultHandle, CUBLAS_OP_N, a.M, a.N, 1, a.fp, a.M, v.fp, 1, 0, result.fp, 1)

# BLAS level 3 operations

proc `*`*[A: SomeReal](a, b: CudaMatrix[A]): CudaMatrix[A] {. inline .} =
  checkDim(a.N == b.M)
  init(result, a.M, b.N)
  check cublasGemm(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_N, a.M, b.N, a.N, 1,
    a.fp, a.M, b.fp, a.N, 0, result.fp, a.M)

# Comparison

template compareApprox(a, b: CudaVector or CudaMatrix): bool =
  const epsilon = 0.000001
  let
    aNorm = l_1(a)
    bNorm = l_1(b)
    dNorm = l_1(a - b)
  (dNorm / (aNorm + bNorm)) < epsilon

proc `=~`*[A: SomeReal](v, w: CudaVector[A]): bool = compareApprox(v, w)

proc `=~`*[A: SomeReal](v, w: CudaMatrix[A]): bool = compareApprox(v, w)

template `!=~`*(a, b: CudaVector or CudaMatrix): bool = not (a =~ b)

# Cloning and slicing

proc clone*[A](v: CudaVector[A]): CudaVector[A] =
  init(result, v.N)
  check cudaMemcpy(result.fp, v.fp, v.N * sizeof(A), cudaMemcpyDeviceToDevice)

proc clone*[A](m: CudaMatrix[A]): CudaMatrix[A] =
  init(result, m.M, m.N)
  check cudaMemcpy(result.fp, m.fp, m.M * m.N * sizeof(A), cudaMemcpyDeviceToDevice)

type
  CArray{.unchecked.}[T] = array[1, T]
  CPointer[T] = ptr CArray[T]

proc plus[A](p: ptr A, n: int): ptr A {.inline.} =
  addr(cast[CPointer[A]](p)[n])

proc `[]`*[A](v: CudaVector[A], s: Slice[int]): CudaVector[A] =
  assert s.a >= 0
  assert s.b < v.N
  let L = s.b - s.a + 1
  init(result, L)
  check cudaMemcpy(result.fp, v.fp.plus(s.a), L * sizeof(A), cudaMemcpyDeviceToDevice)

proc `[]`*[A](m: CudaMatrix[A], s: Slice[int]): CudaMatrix[A] =
  assert s.a >= 0
  assert s.b < m.N
  let L = s.b - s.a + 1
  init(result, m.M, L)
  check cudaMemcpy(result.fp, m.fp.plus(s.a * m.M), m.M * L * sizeof(A), cudaMemcpyDeviceToDevice)