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

type
  CArray{.unchecked.}[T] = array[1, T]
  CPointer[T] = ptr CArray[T]

proc first[T](p: CPointer[T]): ptr T {.inline.} = addr(p[0])

proc first[T](a: var seq[T]): ptr T {.inline.} = addr(a[0])

type
  Complex*[A] = tuple[re, im: A]
  Number* = float32 or float64 or Complex[float32] or Complex[float64]
  SparseMatrixKind* = enum
    CSR, CSC, COO
  SparseMatrixObj*[A] = object
    kind*: SparseMatrixKind
    M*, N*, nnz*: int32
    rows*, cols*: ptr int32
    vals*: ptr A
  SparseMatrix*[A] = ref SparseMatrixObj[A]

proc dealloc*[A](m: SparseMatrix[A]) =
  if m.rows != nil: dealloc(m.rows)
  if m.cols != nil: dealloc(m.cols)
  if m.vals != nil: dealloc(m.vals)

proc rowLen*(m: SparseMatrix): int32 =
  case m.kind
  of CSR: m.M + 1
  of CSC, COO: m.nnz

proc colLen*(m: SparseMatrix): int32 =
  case m.kind
  of CSR, COO: m.nnz
  of CSC: m.N + 1

proc sizes*[A](m: SparseMatrix[A]): tuple[r, c, v: int32] =
  (
    (m.rowLen * sizeof(int32)).int32,
    (m.colLen * sizeof(int32)).int32,
    (m.nnz * sizeof(A)).int32
  )

proc sparse*[A: Number](kind: SparseMatrixKind, M, N, nnz: int32, rows, cols: var seq[int32], vals: var seq[A]): SparseMatrix[A] =
  new result, dealloc
  result.kind = kind
  result.M = N
  result.N = N
  result.nnz = nnz
  let (r, c, v) = result.sizes
  result.rows = cast[ptr int32](alloc(r))
  result.cols = cast[ptr int32](alloc(c))
  result.vals = cast[ptr A](alloc(v))
  copyMem(result.rows, rows.first, r)
  copyMem(result.cols, cols.first, c)
  copyMem(result.vals, vals.first, v)

proc csr*[A: Number](rows, cols: var seq[int32], vals: var seq[A], numCols: int32): SparseMatrix[A] =
  sparse(CSR, rows.len.int32, numCols, vals.len.int32, rows, cols, vals)

proc csc*[A: Number](rows, cols: var seq[int32], vals: var seq[A], numRows: int32): SparseMatrix[A] =
  sparse(CSC, numRows, cols.len.int32, vals.len.int32, rows, cols, vals)

proc coo*[A: Number](rows, cols: var seq[int32], vals: var seq[A], numRows, numCols: int32): SparseMatrix[A] =
  sparse(COO, numRows, numCols, vals.len.int32, rows, cols, vals)