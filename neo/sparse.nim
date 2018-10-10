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
import ./core, ./dense

export core

type
  SparseVector*[A] = ref object
    N*: int32
    indices*: seq[int32]
    vals*: seq[A]
  SparseMatrixKind* = enum
    CSR, CSC, COO
  SparseMatrixObj*[A] = object
    kind*: SparseMatrixKind
    M*, N*, nnz*: int32
    rows*, cols*: seq[int32]
    vals*: seq[A]
  SparseMatrix*[A] = ref SparseMatrixObj[A]

proc len*(v: SparseVector): int {.inline.} = v.N.int

proc nnz*(v: SparseVector): int32 {.inline.} = v.indices.len.int32

proc rowLen*(m: SparseMatrix): int32 =
  case m.kind
  of CSR: m.M + 1
  of CSC, COO: m.nnz

proc colLen*(m: SparseMatrix): int32 =
  case m.kind
  of CSR, COO: m.nnz
  of CSC: m.N + 1

# Initializers

proc sparseVector*[A](N: int32, indices: seq[int32], vals: seq[A]): SparseVector[A] =
  SparseVector[A](N: N, indices: indices, vals: vals)

proc sparseMatrix*[A: Scalar](kind: SparseMatrixKind, M, N, nnz: int32, rows, cols: seq[int32], vals: seq[A]): SparseMatrix[A] =
  SparseMatrix[A](
    kind: kind,
    M: M,
    N: N,
    nnz: nnz,
    rows: rows,
    cols: cols,
    vals: vals
  )

proc csr*[A: Scalar](rows, cols: seq[int32], vals: seq[A], numCols: int32): SparseMatrix[A] =
  sparseMatrix(CSR, rows.len.int32 - 1, numCols, vals.len.int32, rows, cols, vals)

proc csc*[A: Scalar](rows, cols: seq[int32], vals: seq[A], numRows: int32): SparseMatrix[A] =
  sparseMatrix(CSC, numRows, cols.len.int32 - 1, vals.len.int32, rows, cols, vals)

proc coo*[A: Scalar](rows, cols: seq[int32], vals: seq[A], numRows, numCols: int32): SparseMatrix[A] =
  sparseMatrix(COO, numRows, numCols, vals.len.int32, rows, cols, vals)

# Iterators

iterator items*[A](v: SparseVector[A]): A =
  var
    zero: A
    next = v.indices[0]
    count = 0
  for i in 0 ..< v.N:
    if i == next:
      yield v.vals[count]
      inc count
      if count < v.indices.len:
        next = v.indices[count]
    else:
      yield zero

iterator pairs*[A](v: SparseVector[A]): tuple[key: int32, val: A] =
  var
    zero: A
    next = v.indices[0]
    count = 0
  for i in 0'i32 ..< v.N:
    if i == next:
      yield (i, v.vals[count])
      inc count
      if count < v.indices.len:
        next = v.indices[count]
    else:
      yield (i, zero)

iterator nonzero*[A](v: SparseVector[A]): tuple[key: int32, val: A] =
  for i, j in v.indices:
    yield (j, v.vals[i])


iterator items*[A](m: SparseMatrix[A]): A =
  var count = 0
  var zero: A
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
          yield zero
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
          yield zero
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
          yield zero

iterator pairs*[A](m: SparseMatrix[A]): tuple[key: (int32, int32), val: A] =
  var count = 0
  var zero: A
  case m.kind
  of CSR:
    var next = m.cols[0]
    for i in 0'i32 ..< m.M:
      let max = m.rows[i + 1]
      for j in 0'i32 ..< m.N:
        if count < max and j == next:
          yield ((i, j), m.vals[count])
          inc count
          if count < m.nnz:
            next = m.cols[count]
        else:
          yield ((i, j), zero)
  of CSC:
    var next = m.rows[0]
    for j in 0'i32 ..< m.N:
      let max = m.cols[j + 1]
      for i in 0'i32 ..< m.M:
        if count < max and i == next:
          yield ((i, j), m.vals[count])
          inc count
          if count < m.nnz:
            next = m.rows[count]
        else:
          yield ((i, j), zero)
  of COO:
    var
      nextR = m.rows[0]
      nextC = m.cols[0]
    for i in 0'i32 ..< m.M:
      for j in 0'i32 ..< m.N:
        if i == nextR and j == nextC:
          yield ((i, j), m.vals[count])
          inc count
          if count < m.nnz:
            nextR = m.rows[count]
            nextC = m.cols[count]
        else:
          yield ((i, j), zero)

iterator nonzero*[A](m: SparseMatrix[A]): tuple[key: (int32, int32), val: A] =
  case m.kind
  of CSR:
    var
      count = 0
      row = 0'i32
    while count < m.nnz:
      while count >= m.rows[row + 1]:
        inc row
      let n = m.rows[row + 1] - count
      for _ in 1 .. n:
        yield ((row, m.cols[count]), m.vals[count])
        inc count
  of CSC:
    var
      count = 0
      col = 0'i32
    while count < m.nnz:
      while count >= m.cols[col + 1]:
        inc col
      let n = m.cols[col + 1] - count
      for _ in 1 .. n:
        yield ((m.rows[count], col), m.vals[count])
        inc count
  of COO:
    let L = m.rows.len
    for k in 0 ..< L:
      yield ((m.rows[k], m.cols[k]), m.vals[k])

# Conversions

proc dense*[A](v: SparseVector[A]): Vector[A] =
  result = zeros(v.N, A)
  for i, x in v.nonzero:
    result[i] = x

proc dense*[A](m: SparseMatrix[A], order = colMajor): Matrix[A] =
  result = zeros(m.M, m.N, A, order)
  for t, x in m.nonzero:
    let (i, j) = t
    result[i, j] = x

# Equality

# TODO: implement a faster way to check equality
proc `==`*[A](v, w: SparseVector[A]): bool = v.dense == w.dense

proc `==`*[A](m, n: SparseMatrix[A]): bool = m.dense == n.dense

# Printing

proc `$`*[A](v: SparseVector[A]): string = $(v.dense)

proc `$`*[A](m: SparseMatrix[A]): string = $(m.dense)