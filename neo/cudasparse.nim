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

import sequtils
import nimcuda/[cuda_runtime_api, driver_types, cusparse, nimcuda]
import ./sparse

template pointerTo(x: untyped) = cast[ptr pointer](addr x)

proc first[T](a: var seq[T]): ptr T {.inline.} = addr(a[0])

type
  CudaSparseVectorObj*[A] = object
    N*, nnz*: int32
    indices*: ptr int32
    vals*: ptr A
  CudaSparseVector*[A] = ref CudaSparseVectorObj[A]
  CudaSparseMatrixObj*[A] = object
    kind*: SparseMatrixKind
    M*, N*, nnz*: int32
    rows*, cols*: ptr int32
    vals*: ptr A
  CudaSparseMatrix*[A] = ref CudaSparseMatrixObj[A]

proc dealloc*[A](v: CudaSparseVector[A]) =
  if v.indices != nil: check cudaFree(v.indices)
  if v.vals != nil: check cudaFree(v.vals)

proc dealloc*[A](m: CudaSparseMatrix[A]) =
  if m.rows != nil: check cudaFree(m.rows)
  if m.cols != nil: check cudaFree(m.cols)
  if m.vals != nil: check cudaFree(m.vals)

proc rowLen*(m: CudaSparseMatrix): int32 =
  case m.kind
  of CSR: m.M + 1
  of CSC, COO: m.nnz

proc colLen*(m: CudaSparseMatrix): int32 =
  case m.kind
  of CSR, COO: m.nnz
  of CSC: m.N + 1

proc sizes[A](m: SparseVector[A] or CudaSparseVector[A]): tuple[i, v: int32] =
  (
    (m.N * sizeof(int32)).int32,
    (m.nnz * sizeof(A)).int32
  )

proc sizes[A](m: SparseMatrix[A] or CudaSparseMatrix[A]): tuple[r, c, v: int32] =
  (
    (m.rowLen * sizeof(int32)).int32,
    (m.colLen * sizeof(int32)).int32,
    (m.nnz * sizeof(A)).int32
  )

var defaultHandle: cusparseHandle_t
check cusparseCreate(addr defaultHandle)

const idx = CUSPARSE_INDEX_BASE_ZERO

template allocateAll(x, r, c, v: untyped) =
  check cudaMalloc(pointerTo x.rows, r)
  check cudaMalloc(pointerTo x.cols, c)
  check cudaMalloc(pointerTo x.vals, v)

proc gpu*[A: Number](v: SparseVector[A]): CudaSparseVector[A] =
  new result, dealloc
  result.N = v.N
  result.nnz = v.nnz
  let (iLen, vLen) = v.sizes
  check cudaMalloc(pointerTo result.indices, iLen)
  check cudaMalloc(pointerTo result.vals, vLen)
  check cudaMemcpy(result.indices, v.indices.first, iLen, cudaMemcpyHostToDevice)
  check cudaMemcpy(result.vals, v.vals.first, vLen, cudaMemcpyHostToDevice)

proc gpu*[A: Number](m: SparseMatrix[A]): CudaSparseMatrix[A] =
  new result, dealloc
  result.kind = m.kind
  result.M = m.M
  result.N = m.N
  result.nnz = m.nnz
  let (r, c, v) = result.sizes
  allocateAll(result, r, c, v)
  check cudaMemcpy(result.rows, m.rows.first, r, cudaMemcpyHostToDevice)
  check cudaMemcpy(result.cols, m.cols.first, c, cudaMemcpyHostToDevice)
  check cudaMemcpy(result.vals, m.vals.first, v, cudaMemcpyHostToDevice)

proc cpu*[A: Number](v: CudaSparseVector[A]): SparseVector[A] =
  result = SparseVector[A](
    N: v.N,
    indices: newSeq[int32](v.nnz),
    vals: newSeq[A](v.nnz)
  )
  let (iLen, vLen) = v.sizes
  check cudaMemcpy(result.indices.first, v.indices, iLen, cudaMemcpyDeviceToHost)
  check cudaMemcpy(result.vals.first, v.vals, vLen, cudaMemcpyDeviceToHost)

proc cpu*[A: Number](m: CudaSparseMatrix[A]): SparseMatrix[A] =
  new result
  result.kind = m.kind
  result.M = m.M
  result.N = m.N
  result.nnz = m.nnz
  let (r, c, v) = result.sizes
  result.rows = newSeq[int32](m.rowLen)
  result.cols = newSeq[int32](m.colLen)
  result.vals = newSeq[A](m.nnz)
  check cudaMemcpy(result.rows.first, m.rows, r, cudaMemcpyDeviceToHost)
  check cudaMemcpy(result.cols.first, m.cols, c, cudaMemcpyDeviceToHost)
  check cudaMemcpy(result.vals.first, m.vals, v, cudaMemcpyDeviceToHost)

proc toCsr*[A: Number](m: CudaSparseMatrix[A], handle = defaultHandle): CudaSparseMatrix[A] =
  new result, dealloc
  result.kind = CSR
  result.M = m.M
  result.N = m.N
  result.nnz = m.nnz
  let (r, c, v) = result.sizes
  allocateAll(result, r, c, v)
  case m.kind
  of CSR:
    check cudaMemcpy(result.rows, m.rows, r, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)
  of CSC:
    when A is float32:
      check cusparseScsr2csc(handle, m.N, m.M, m.nnz, result.vals, result.cols, result.rows, m.vals, m.cols, m.rows, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is float64:
      check cusparseDcsr2csc(handle, m.N, m.M, m.nnz, result.vals, result.cols, result.rows, m.vals, m.cols, m.rows, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is Complex[float32]:
      check cusparseCcsr2csc(handle, m.N, m.M, m.nnz, result.vals, result.cols, result.rows, m.vals, m.cols, m.rows, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is Complex[float64]:
      check cusparseZcsr2csc(handle, m.N, m.M, m.nnz, result.vals, result.cols, result.rows, m.vals, m.cols, m.rows, CUSPARSE_ACTION_NUMERIC, idx)
  of COO:
    check cusparseXcoo2csr(handle, m.rows, m.nnz, m.N, result.rows, idx)
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)

proc toCsc*[A: Number](m: CudaSparseMatrix[A], handle = defaultHandle): CudaSparseMatrix[A] =
  case m.kind
  of CSR:
    new result, dealloc
    result.kind = CSC
    result.M = m.M
    result.N = m.N
    result.nnz = m.nnz
    let (r, c, v) = result.sizes
    allocateAll(result, r, c, v)
    when A is float32:
      check cusparseScsr2csc(handle, m.M, m.N, m.nnz, result.vals, result.rows, result.cols, m.vals, m.rows, m.cols, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is float64:
      check cusparseDcsr2csc(handle, m.M, m.N, m.nnz, result.vals, result.rows, result.cols, m.vals, m.rows, m.cols, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is Complex[float32]:
      check cusparseCcsr2csc(handle, m.M, m.N, m.nnz, result.vals, result.rows, result.cols, m.vals, m.rows, m.cols, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is Complex[float64]:
      check cusparseZcsr2csc(handle, m.M, m.N, m.nnz, result.vals, result.rows, result.cols, m.vals, m.rows, m.cols, CUSPARSE_ACTION_NUMERIC, idx)
  of CSC:
    new result, dealloc
    result.kind = CSC
    result.M = m.M
    result.N = m.N
    result.nnz = m.nnz
    let (r, c, v) = result.sizes
    allocateAll(result, r, c, v)
    check cudaMemcpy(result.rows, m.rows, r, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)
  of COO:
    result = m.toCsr().toCsc()

proc toCoo*[A: Number](m: CudaSparseMatrix[A], handle = defaultHandle): CudaSparseMatrix[A] =
  new result, dealloc
  result.kind = COO
  result.M = m.M
  result.N = m.N
  result.nnz = m.nnz
  let (r, c, v) = result.sizes
  allocateAll(result, r, c, v)
  case m.kind
  of CSR:
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)
    check cusparseXcsr2coo(handle, m.rows, m.nnz, m.M, result.rows, idx)
  of CSC:
    result = m.toCsr().toCoo()
  of COO:
    check cudaMemcpy(result.rows, m.rows, r, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)