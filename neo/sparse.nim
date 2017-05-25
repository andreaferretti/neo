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
import ./dense

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
    rows*, cols*: seq[int32]
    vals*: seq[A]
  SparseMatrix*[A] = ref SparseMatrixObj[A]

proc rowLen*(m: SparseMatrix): int32 =
  case m.kind
  of CSR: m.M + 1
  of CSC, COO: m.nnz

proc colLen*(m: SparseMatrix): int32 =
  case m.kind
  of CSR, COO: m.nnz
  of CSC: m.N + 1

# Initializers

proc sparse*[A: Number](kind: SparseMatrixKind, M, N, nnz: int32, rows, cols: seq[int32], vals: seq[A]): SparseMatrix[A] =
  SparseMatrix[A](
    kind: kind,
    M: M,
    N: N,
    nnz: nnz,
    rows: rows,
    cols: cols,
    vals: vals
  )

proc csr*[A: Number](rows, cols: seq[int32], vals: seq[A], numCols: int32): SparseMatrix[A] =
  sparse(CSR, rows.len.int32 - 1, numCols, vals.len.int32, rows, cols, vals)

proc csc*[A: Number](rows, cols: seq[int32], vals: seq[A], numRows: int32): SparseMatrix[A] =
  sparse(CSC, numRows, cols.len.int32 - 1, vals.len.int32, rows, cols, vals)

proc coo*[A: Number](rows, cols: seq[int32], vals: seq[A], numRows, numCols: int32): SparseMatrix[A] =
  sparse(COO, numRows, numCols, vals.len.int32, rows, cols, vals)

# Iterators

iterator items*[A](m: SparseMatrix[A]): A =
  var count = 0
  case m.kind
  of CSR:
    var next = m.cols[0]
    for i in 0 ..< m.M:
      let max = m.rows[i + 1]
      for j in 0 ..< m.N:
        if count < max and j == next:
          yield m.vals[count]
          inc count
          if count < m.nnz:
            next = m.cols[count]
        else:
          yield 0
  of CSC:
    var next = m.rows[0]
    for j in 0 ..< m.N:
      let max = m.cols[j + 1]
      for i in 0 ..< m.M:
        if count < max and i == next:
          yield m.vals[count]
          inc count
          if count < m.nnz:
            next = m.rows[count]
        else:
          yield 0
  of COO:
    var
      nextR = m.rows[0]
      nextC = m.cols[0]
    for i in 0 ..< m.M:
      for j in 0 ..< m.N:
        if i == nextR and j == nextC:
          yield m.vals[count]
          inc count
          if count < m.nnz:
            nextR = m.rows[count]
            nextC = m.cols[count]
        else:
          yield 0

# Conversions

proc dense*[A](m: SparseMatrix[A]): Matrix[A] =
  result = Matrix[A](
    order: (if m.kind == CSC: colMajor else: rowMajor),
    M: m.M,
    N: m.N,
    data: newSeq[A](m.M * m.N)
  )
  var i = 0
  for x in m:
    result.data[i] = x
    inc i

# Equality

# TODO: implement a faster way to check equality
proc `==`*[A](m, n: SparseMatrix[A]): bool = m.dense == n.dense