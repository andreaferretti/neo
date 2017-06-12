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

proc freeDeviceMemory[A: SomeReal](p: ref[ptr A]) =
  check cudaFree(p[])

proc isContiguous*(v: CudaVector): bool {.inline.} =
  v.step == 1

proc isContiguous*(m: CudaMatrix): bool {.inline.} =
  if m.order == colMajor: m.M == m.ld
  else: m.N == m.ld

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

proc gpu*[A: SomeReal](v: Vector[A]): CudaVector[A] =
  init(result, v.len)
  check cublasSetVector(v.len.int32, sizeof(A).int32, v.fp, v.step.int32, result.fp, result.step)

proc gpu*[A: SomeReal](m: Matrix[A]): CudaMatrix[A] =
  if m.order == rowMajor:
    raise newException(ValueError, "m must be column major")
  init(result, m.M, m.N)
  check cublasSetMatrix(m.M.int32, m.N.int32, sizeof(A).int32, m.fp, m.ld.int32, result.fp, result.ld)

proc cpu*[A: SomeReal](v: CudaVector[A]): Vector[A] =
  result = zeros(v.len, A)
  check cublasGetVector(v.len, sizeof(A).int32, v.fp, v.step, result.fp, result.step.int32)

proc cpu*[A: SomeReal](m: CudaMatrix[A]): Matrix[A] =
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

  # Slicing

proc `[]`*[A](v: CudaVector[A], s: Slice[int]): CudaVector[A] =
  checkBounds(s.a >= 0 and s.b < v.len)
  let vp = cast[CPointer[A]](v.fp)
  let L = s.b - s.a + 1
  init(result, L)
  check cudaMemcpy(result.fp, addr(vp[s.a]), L * sizeof(A), cudaMemcpyDeviceToDevice)

proc `[]`*[A](m: CudaMatrix[A], s: Slice[int]): CudaMatrix[A] =
  checkBounds(s.a >= 0 and s.b < m.N)
  let mp = cast[CPointer[A]](m.fp)
  let L = s.b - s.a + 1
  init(result, m.M, L)
  check cudaMemcpy(result.fp, addr(mp[s.a * m.M]), m.M * L * sizeof(A), cudaMemcpyDeviceToDevice)

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

# BLAS level 1 operations

proc `*=`*[A: SomeReal](v: var CudaVector[A], k: A) {. inline .} =
  var k1 = k
  check scal(defaultHandle, v.len, addr(k1), v.fp, v.step)

proc `*`*[A: SomeReal](v: CudaVector[A], k: A): CudaVector[A]  {. inline .} =
  init(result, v.len)
  check copy(defaultHandle, v.len, v.fp, v.step, result.fp, result.step)
  result *= k

proc `+=`*[A: SomeReal](v: var CudaVector[A], w: CudaVector[A]) {. inline .} =
  checkDim(v.len == w.len)
  var alpha: A = 1
  check axpy(defaultHandle, v.len, addr(alpha), w.fp, w.step, v.fp, v.step)

proc `+`*[A: SomeReal](v, w: CudaVector[A]): CudaVector[A] {. inline .} =
  checkDim(v.len == w.len)
  init(result, v.len)
  check copy(defaultHandle, v.len, v.fp, v.step, result.fp, result.step)
  result += w

proc `-=`*[A: SomeReal](v: var CudaVector[A], w: CudaVector[A]) {. inline .} =
  checkDim(v.len == w.len)
  var alpha: A = -1
  check axpy(defaultHandle, v.len, addr(alpha), w.fp, w.step, v.fp, v.step)

proc `-`*[A: SomeReal](v, w: CudaVector[A]): CudaVector[A] {. inline .} =
  checkDim(v.len == w.len)
  init(result, v.len)
  check copy(defaultHandle, v.len, v.fp, v.step, result.fp, result.step)
  result -= w

proc `*`*[A: SomeReal](v, w: CudaVector[A]): A {. inline .} =
  checkDim(v.len == w.len)
  check dot(defaultHandle, v.len, v.fp, v.step, w.fp, w.step, addr(result))

proc l_2*[A: SomeReal](v: CudaVector[A]): A {. inline .} =
  check nrm2(defaultHandle, v.len, v.fp, v.step, addr(result))

proc l_1*[A: SomeReal](v: CudaVector[A]): A {. inline .} =
  check asum(defaultHandle, v.len, v.fp, v.step, addr(result))

proc `*=`*[A: SomeReal](m: var CudaMatrix[A], k: A) {. inline .} =
  var k1 = k
  check scal(defaultHandle, m.M * m.N, addr(k1), m.fp, 1)

proc `*`*[A: SomeReal](m: CudaMatrix[A], k: A): CudaMatrix[A]  {. inline .} =
  init(result, m.M, m.N)
  check copy(defaultHandle, m.M * m.N, m.fp, 1, result.fp, 1)
  result *= k

template `*`*[A: SomeReal](k: A, v: CudaVector[A] or CudaMatrix[A]): auto =
  v * k

template `/`*[A: SomeReal](v: CudaVector[A] or CudaMatrix[A], k: A): auto =
  v * (1 / k)

template `/=`*[A: SomeReal](v: var CudaVector[A] or var CudaMatrix[A], k: A) =
  v *= (1 / k)

proc `+=`*[A: SomeReal](a: var CudaMatrix[A], b: CudaMatrix[A]) {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  var alpha: A = 1
  check axpy(defaultHandle, a.M * a.N, addr(alpha), b.fp, 1, a.fp, 1)

proc `+`*[A: SomeReal](a, b: CudaMatrix[A]): CudaMatrix[A]  {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  init(result, a.M, a.N)
  check copy(defaultHandle, a.M * a.N, a.fp, 1, result.fp, 1)
  result += b

proc `-=`*[A: SomeReal](a: var CudaMatrix[A], b: CudaMatrix[A]) {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  var alpha: A = -1
  check axpy(defaultHandle, a.M * a.N, addr(alpha), b.fp, 1, a.fp, 1)

proc `-`*[A: SomeReal](a, b: CudaMatrix[A]): CudaMatrix[A]  {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  init(result, a.M, a.N)
  check copy(defaultHandle, a.M * a.N, a.fp, 1, result.fp, 1)
  result -= b

proc l_2*[A: SomeReal](m: CudaMatrix[A]): A {. inline .} =
  check nrm2(defaultHandle, m.M * m.N, m.fp, 1, addr(result))

proc l_1*[A: SomeReal](m: CudaMatrix[A]): A {. inline .} =
  check asum(defaultHandle, m.M * m.N, m.fp, 1, addr(result))

# BLAS level 2 operations

proc `*`*[A: SomeReal](a: CudaMatrix[A], v: CudaVector[A]): CudaVector[A]  {. inline .} =
  checkDim(a.N == v.len)
  init(result, a.M)
  var
    alpha: A = 1
    beta: A = 0
  check gemv(defaultHandle, CUBLAS_OP_N, a.M, a.N, addr(alpha), a.fp, a.M,
    v.fp, v.step, addr(beta), result.fp, 1)

# BLAS level 3 operations

proc `*`*[A: SomeReal](a, b: CudaMatrix[A]): CudaMatrix[A] {. inline .} =
  checkDim(a.N == b.M)
  init(result, a.M, b.N)
  var
    alpha: A = 1
    beta: A = 0
  check gemm(defaultHandle, CUBLAS_OP_N, CUBLAS_OP_N, a.M, b.N, a.N,
    addr(alpha), a.fp, a.M, b.fp, a.N, addr(beta), result.fp, a.M)

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