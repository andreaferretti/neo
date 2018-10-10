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
export core

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

type
  Array32[N: static[int]] = array[N, float32]
  Array64[N: static[int]] = array[N, float64]
  DoubleArray32[M, N: static[int]] = array[M, array[N, float32]]
  DoubleArray64[M, N: static[int]] = array[M, array[N, float64]]

proc vector*[A](data: seq[A]): Vector[A] =
  result = Vector[A](step: 1, len: data.len)
  shallowCopy(result.data, data)
  result.fp = addr(result.data[0])

proc vector*[A](data: varargs[A]): Vector[A] =
  vector[A](@data)

proc stackVector*[N: static[int]](a: var Array32[N]): Vector[float32] =
  Vector[float32](fp: addr a[0], len: N, step: 1)

proc stackVector*[N: static[int]](a: var Array64[N]): Vector[float64] =
  Vector[float64](fp: addr a[0], len: N, step: 1)

proc sharedVector*(N: int, A: typedesc): Vector[A] =
  Vector[A](fp: cast[ptr A](allocShared0(N * sizeof(A))), len: N, step: 1)

proc dealloc*[A](v: Vector[A]) =
  deallocShared(cast[pointer](v.fp))

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
  makeVectorI[float64](N, rand(max))

proc randomVector*(N: int, max: float32): Vector[float32] =
  makeVectorI[float32](N, rand(max).float32)

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

proc randomMatrix*[A: SomeFloat](M, N: int, max: A = 1, order = colMajor): Matrix[A] =
  result = matrix[A](order, M, N, newSeq[A](M * N))
  for i in 0 ..< (M * N):
    result.data[i] = rand(max)

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
  when compileOption("assertions"):
    for x in xs:
      checkDim(xs[0].len == x.len, "The dimensions do not match")
  makeMatrixIJ(A, xs.len, xs[0].len, xs[i][j], order)

proc matrix*[A](xs: seq[Vector[A]], order = colMajor): Matrix[A] =
  when compileOption("assertions"):
    for x in xs:
      checkDim(xs[0].len == x.len, "The dimensions do not match")
  makeMatrixIJ(A, xs.len, xs[0].len, xs[i][j], order)

proc stackMatrix*[M, N: static[int]](a: var DoubleArray32[M, N], order = colMajor): Matrix[float32] =
  let M1: int = if order == colMajor: N else: M
  let N1: int = if order == colMajor: M else: N
  result = Matrix[float32](
    order: order,
    fp: addr a[0][0],
    M: M1,
    N: N1,
    ld: N
  )

proc stackMatrix*[M, N: static[int]](a: var DoubleArray64[M, N], order = colMajor): Matrix[float64] =
  let M1: int = if order == colMajor: N else: M
  let N1: int = if order == colMajor: M else: N
  Matrix[float64](
    order: order,
    fp: addr a[0][0],
    M: M1,
    N: N1,
    ld: N
  )

proc sharedMatrix*(M, N: int, A: typedesc, order = colMajor): Matrix[A] =
  Matrix[A](
    order: order,
    fp: cast[ptr A](allocShared0(M * N * sizeof(A))),
    M: if order == colMajor: N else: M,
    N: if order == colMajor: M else: N,
    ld: N
  )

proc dealloc*[A](m: Matrix[A]) =
  deallocShared(cast[pointer](m.fp))

proc diag*[A: SomeFloat](xs: varargs[A]): Matrix[A] =
  let n = xs.len
  result = zeros(n, n, A)
  result.shape.incl(Diagonal)
  for i in 0 ..< n:
    result.data[i * (n + 1)] = xs[i]

# Accessors

template elColMajor(ap, a, i, j: untyped): untyped =
  ap[j * a.ld + i]

template elRowMajor(ap, a, i, j: untyped): untyped =
  ap[i * a.ld + j]

proc `[]`*[A](v: Vector[A], i: int): A {. inline .} =
  checkBounds(i >= 0 and i < v.len)
  return cast[CPointer[A]](v.fp)[v.step * i]

proc `[]=`*[A](v: Vector[A], i: int, val: A) {. inline .} =
  checkBounds(i >= 0 and i < v.len)
  cast[CPointer[A]](v.fp)[v.step * i] = val

proc `[]`*[A](m: Matrix[A], i, j: int): A {. inline .} =
  checkBounds(i >= 0 and i < m.M)
  checkBounds(j >= 0 and j < m.N)
  let mp = cast[CPointer[A]](m.fp)
  if m.order == colMajor:
    return elColMajor(mp, m, i, j)
  else:
    return elRowMajor(mp, m, i, j)

proc `[]=`*[A](m: var Matrix[A], i, j: int, val: A) {. inline .} =
  checkBounds(i >= 0 and i < m.M)
  checkBounds(j >= 0 and j < m.N)
  let mp = cast[CPointer[A]](m.fp)
  if m.order == colMajor:
    elColMajor(mp, m, i, j) = val
  else:
    elRowMajor(mp, m, i, j) = val

proc column*[A](m: Matrix[A], j: int): Vector[A] {. inline .} =
  checkBounds(j >= 0 and j < m.N)
  let mp = cast[CPointer[A]](m.fp)
  if m.order == colMajor:
    result = Vector[A](
      data: m.data,
      fp: addr(mp[j * m.ld]),
      len: m.M,
      step: 1
    )
  else:
    result = Vector[A](
      data: m.data,
      fp: addr(mp[j]),
      len: m.M,
      step: m.ld
    )

proc row*[A](m: Matrix[A], i: int): Vector[A] {. inline .} =
  checkBounds(i >= 0 and i < m.M)
  let mp = cast[CPointer[A]](m.fp)
  if m.order == colMajor:
    result = Vector[A](
      data: m.data,
      fp: addr(mp[i]),
      len: m.N,
      step: m.ld
    )
  else:
    result = Vector[A](
      data: m.data,
      fp: addr(mp[i * m.ld]),
      len: m.N,
      step: 1
    )

proc dim*(m: Matrix): tuple[rows, columns: int] = (m.M, m.N)

# Iterators

iterator items*[A](v: Vector[A]): auto {. inline .} =
  let vp = cast[CPointer[A]](v.fp)
  var pos = 0
  for i in 0 ..< v.len:
    yield vp[pos]
    pos += v.step

iterator pairs*[A](v: Vector[A]): auto {. inline .} =
  let vp = cast[CPointer[A]](v.fp)
  var pos = 0
  for i in 0 ..< v.len:
    yield (i, vp[pos])
    pos += v.step

iterator columns*[A](m: Matrix[A]): auto {. inline .} =
  let
    mp = cast[CPointer[A]](m.fp)
    step = if m.order == colMajor: m.ld else: 1
  var v = m.column(0)
  yield v
  for j in 1 ..< m.N:
    v.fp = addr(mp[j * step])
    yield v

iterator rows*[A](m: Matrix[A]): auto {. inline .} =
  let
    mp = cast[CPointer[A]](m.fp)
    step = if m.order == rowMajor: m.ld else: 1
  var v = m.row(0)
  yield v
  for i in 1 ..< m.M:
    v.fp = addr(mp[i * step])
    yield v

iterator columnsSlow*[A](m: Matrix[A]): auto {. inline .} =
  for i in 0 ..< m.N:
    yield m.column(i)

iterator rowsSlow*[A](m: Matrix[A]): auto {. inline .} =
  for i in 0 ..< m.M:
    yield m.row(i)

iterator items*[A](m: Matrix[A]): auto {. inline .} =
  let mp = cast[CPointer[A]](m.fp)
  if m.order == colMajor:
    for i in 0 ..< m.M:
      for j in 0 ..< m.N:
        yield elColMajor(mp, m, i, j)
  else:
    for i in 0 ..< m.M:
      for j in 0 ..< m.N:
        yield elRowMajor(mp, m, i, j)

iterator pairs*[A](m: Matrix[A]): auto {. inline .} =
  let mp = cast[CPointer[A]](m.fp)
  if m.order == colMajor:
    for i in 0 ..< m.M:
      for j in 0 ..< m.N:
        yield ((i, j), elColMajor(mp, m, i, j))
  else:
    for i in 0 ..< m.M:
      for j in 0 ..< m.N:
        yield ((i, j), elRowMajor(mp, m, i, j))

# Conversion

proc clone*[A](v: Vector[A]): Vector[A] =
  if v.isFull:
    var dataCopy = v.data
    return vector(dataCopy)
  else:
    return vector(toSeq(v.items))

proc clone*[A](m: Matrix[A]): Matrix[A] =
  if m.isFull:
    var dataCopy = m.data
    result = matrix[A](data = dataCopy, order = m.order, M = m.M, N = m.N)
  else:
    result = matrix(m.order, m.M, m.N, newSeq[A](m.M * m.N))
    # TODO: copy one row or column at a time
    for t, v in m:
      let (i, j) = t
      result[i, j] = v

proc map*[A, B](v: Vector[A], f: proc(x: A): B): Vector[B] =
  result = zeros(v.len, B)
  for i, x in v:
    result.data[i] = f(x) # `result` is full here, we can assign `data` directly

proc map*[A, B](m: Matrix[A], f: proc(x: A): B): Matrix[B] =
  result = zeros(m.M, m.N, B, m.order)
  if m.isFull:
    for i in 0 ..< (m.M * m.N):
      result.data[i] = f(m.data[i])
  else:
    for t, v in m:
      let (i, j) = t
      # TODO: make things faster here
      result[i, j] = f(v)

proc to32*(v: Vector[float64]): Vector[float32] =
  if v.isFull:
    vector(v.data.mapIt(it.float32))
  else:
    vector(v.mapIt(it.float32))

proc to64*(v: Vector[float32]): Vector[float64] =
  if v.isFull:
    vector(v.data.mapIt(it.float64))
  else:
    vector(v.mapIt(it.float64))

proc to32*(m: Matrix[float64]): Matrix[float32] =
  if m.isFull:
    matrix(data = m.data.mapIt(it.float32), order = m.order, M = m.M, N = m.N)
  else:
    m.map(proc(x: float64): float32 = x.float32)

proc to64*(m: Matrix[float32]): Matrix[float64] =
  if m.isFull:
    matrix(data = m.data.mapIt(it.float64), order = m.order, M = m.M, N = m.N)
  else:
    m.map(proc(x: float32): float64 = x.float64)

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
  if m.isFull:
    matrix(
      order = (if m.order == rowMajor: colMajor else: rowMajor),
      M = m.N,
      N = m.M,
      data = m.data
    )
  else:
    m.clone().t

proc reshape*[A](m: Matrix[A], a, b: int): Matrix[A] =
  if m.isFull:
    checkDim(m.M * m.N == a * b, "The dimensions do not match: M = " & $(m.M) & ", N = " & $(m.N) & ", A = " & $(a) & ", B = " & $(b))
    result = matrix(
      order = m.order,
      M = a,
      N = b,
      data = m.data
    )
  else:
    result = m.clone().reshape(a, b)

proc asMatrix*[A](v: Vector[A], a, b: int, order = colMajor): Matrix[A] =
  if v.isFull:
    checkDim(v.len == a * b, "The dimensions do not match: N = " & $(v.len) & ", A = " & $(a) & ", B = " & $(b))
    result = matrix(
      order = order,
      M = a,
      N = b,
      data = v.data
    )
  else:
    result = v.clone().asMatrix(a, b, order)

proc asVector*[A](m: Matrix[A]): Vector[A] =
  if m.isFull:
    vector(m.data)
  else:
    vector(toSeq(m.items))

# Stacking

proc hstack*[A](vectors: varargs[Vector[A]]): Vector[A] =
  var L = 0
  for v in vectors:
    L += v.len
  result = vector(newSeq[A](L))
  var pos = 0
  for v in vectors:
    for j, x in v:
      result.data[pos + j] = x
    pos += v.len

template concat*[A](vectors: varargs[Vector[A]]): Vector[A] =
  hstack(vectors)

template vstack*[A](vectors: varargs[Vector[A]]): Matrix[A] =
  matrix(@vectors)

proc hstack*[A](matrices: varargs[Matrix[A]]): Matrix[A] =
  let M = matrices[0].M
  when compileOption("assertions"):
    for m in matrices:
      checkDim(M == m.M, "The vertical dimensions do not match")
  var N = 0
  for m in matrices:
    N += m.N
  result = matrix[A](colMajor, M, N, newSeq[A](M * N))
  var pos = 0
  for m in matrices:
    for i in 0 ..< m.M:
      for j in 0 ..< m.N:
        result[i, pos + j] = m[i, j]
    pos += m.N

proc vstack*[A](matrices: varargs[Matrix[A]]): Matrix[A] =
  let N = matrices[0].N
  when compileOption("assertions"):
    for m in matrices:
      checkDim(N == m.N, "The horizontal dimensions do not match")
  var M = 0
  for m in matrices:
    M += m.M
  result = matrix[A](colMajor, M, N, newSeq[A](M * N))
  var pos = 0
  for m in matrices:
    for i in 0 ..< m.M:
      for j in 0 ..< m.N:
        result[pos + i, j] = m[i, j]
    pos += m.M

# Slice accessors

proc pointerAt[A](v: Vector[A], i: int): ptr A {. inline .} =
  let s = cast[CPointer[A]](v.fp)
  addr s[i]

proc `[]`*[A](v: Vector[A], s: Slice[int]): Vector[A] {. inline .} =
  Vector[A](data: v.data, fp: v.pointerAt(s.a), step: v.step, len: s.b - s.a + 1)

proc `[]=`*[A](v: var Vector[A], s: Slice[int], val: Vector[A]) {. inline .} =
  checkBounds(s.a >= 0 and s.b < v.len)
  checkDim(s.b - s.a + 1 == val.len)
  when A is SomeFloat:
    copy(val.len, val.fp, val.step, v.pointerAt(s.a), v.step)
  else:
    var count = 0
    for i in s:
      v[i] = val[count]
      count += 1

type All* = object

proc `[]`*[A](m: Matrix[A], rows, cols: Slice[int]): Matrix[A] =
  checkBounds(rows.a >= 0 and rows.b < m.M)
  checkBounds(cols.a >= 0 and cols.b < m.N)
  let
    mp = cast[CPointer[A]](m.fp)
    fp =
      if m.order == colMajor:
        addr(elColMajor(mp, m, rows.a, cols.a))
      else:
        addr(elRowMajor(mp, m, rows.a, cols.a))
  result = Matrix[A](
    order: m.order,
    M: (rows.b - rows.a + 1),
    N: (cols.b - cols.a + 1),
    ld: m.ld,
    data: m.data,
    fp: fp
  )

proc `[]`*[A](m: Matrix[A], rows: Slice[int], cols: typedesc[All]): Matrix[A] =
  m[rows, 0 ..< m.N]

proc `[]`*[A](m: Matrix[A], rows: typedesc[All], cols: Slice[int]): Matrix[A] =
  m[0 ..< m.M, cols]

proc `[]=`*[A](m: var Matrix[A], rows, cols: Slice[int], val: Matrix[A]) {. inline .} =
  checkBounds(rows.a >= 0 and rows.b < m.M)
  checkBounds(cols.a >= 0 and cols.b < m.N)
  checkDim(rows.len == val.M)
  checkDim(cols.len == val.N)
  let
    mp = cast[CPointer[A]](m.fp)
    vp = cast[CPointer[A]](val.fp)
  when A is SomeFloat:
    if m.order == colMajor:
      if val.order == colMajor:
        var col = 0
        for c in cols:
          copy(val.M, addr elColMajor(vp, val, 0, col), 1, addr elColMajor(mp, m, rows.a, c), 1)
          col += 1
      else:
        var col = 0
        for c in cols:
          copy(val.M, addr elRowMajor(vp, val, 0, col), val.ld, addr elColMajor(mp, m, rows.a, c), 1)
          col += 1
    else:
      if val.order == colMajor:
        var row = 0
        for r in rows:
          copy(val.N, addr elColMajor(vp, val, row, 0), val.ld, addr elRowMajor(mp, m, r, cols.a), 1)
          row += 1
      else:
        var row = 0
        for r in rows:
          copy(val.N, addr elRowMajor(vp, val, row, 0), 1, addr elRowMajor(mp, m, r, cols.a), 1)
          row += 1
  else:
    var row = 0
    var col = 0
    if m.order == colMajor:
      if val.order == colMajor:
        for r in rows:
          col = 0
          for c in cols:
            elColMajor(mp, m, r, c) = elColMajor(vp, val, row, col)
            col += 1
          row += 1
      else:
        for r in rows:
          col = 0
          for c in cols:
            elColMajor(mp, m, r, c) = elRowMajor(vp, val, row, col)
            col += 1
          row += 1
    else:
      if val.order == colMajor:
        for r in rows:
          col = 0
          for c in cols:
            elRowMajor(mp, m, r, c) = elColMajor(vp, val, row, col)
            col += 1
          row += 1
      else:
        for r in rows:
          col = 0
          for c in cols:
            elRowMajor(mp, m, r, c) = elRowMajor(vp, val, row, col)
            col += 1
          row += 1

proc `[]=`*[A](m: var Matrix[A], rows: Slice[int], cols: typedesc[All], val: Matrix[A]) {. inline .} =
  m[rows, 0 ..< m.N] = val

proc `[]=`*[A](m: var Matrix[A], rows: typedesc[All], cols: Slice[int], val: Matrix[A]) {. inline .} =
  m[0 ..< m.M, cols] = val

# BLAS level 1 operations

proc `*=`*[A: SomeFloat](v: var Vector[A], k: A) {. inline .} =
  scal(v.len, k, v.fp, v.step)

proc `*`*[A: SomeFloat](v: Vector[A], k: A): Vector[A] {. inline .} =
  let N = v.len
  result = vector(newSeq[A](N))
  copy(N, v.fp, v.step, result.fp, result.step)
  scal(N, k, result.fp, result.step)

proc `+=`*[A: SomeFloat](v: var Vector[A], w: Vector[A]) {. inline .} =
  checkDim(v.len == w.len)
  let N = v.len
  axpy(N, 1, w.fp, w.step, v.fp, v.step)

proc `+`*[A: SomeFloat](v, w: Vector[A]): Vector[A]  {. inline .} =
  checkDim(v.len == w.len)
  let N = v.len
  result = vector(newSeq[A](N))
  copy(N, v.fp, v.step, result.fp, result.step)
  axpy(N, 1, w.fp, w.step, result.fp, result.step)

proc `-=`*[A: SomeFloat](v: var Vector[A], w: Vector[A]) {. inline .} =
  checkDim(v.len == w.len)
  let N = v.len
  axpy(N, -1, w.fp, w.step, v.fp, v.step)

proc `-`*[A: SomeFloat](v, w: Vector[A]): Vector[A]  {. inline .} =
  checkDim(v.len == w.len)
  let N = v.len
  result = vector(newSeq[A](N))
  copy(N, v.fp, v.step, result.fp, result.step)
  axpy(N, -1, w.fp, w.step, result.fp, result.step)

proc `*`*[A: SomeFloat](v, w: Vector[A]): A {. inline .} =
  checkDim(v.len == w.len)
  return dot(v.len, v.fp, v.step, w.fp, w.step)

proc l_2*[A: SomeFloat](v: Vector[A]): auto {. inline .} =
  nrm2(v.len, v.fp, v.step)

proc l_1*[A: SomeFloat](v: Vector[A]): auto {. inline .} =
  asum(v.len, v.fp, v.step)

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

proc `*=`*[A: SomeFloat](m: var Matrix[A], k: A) {. inline .} =
  if m.isFull:
    scal(m.M * m.N, k, m.fp, 1)
  else:
    if m.order == colMajor:
      for c in m.columns:
        scal(m.M, k, c.fp, c.step)
    else:
      for r in m.rows:
        scal(m.N, k, r.fp, r.step)

proc `*`*[A: SomeFloat](m: Matrix[A], k: A): Matrix[A]  {. inline .} =
  if m.isFull:
    result.initLike(m)
    copy(m.len, m.fp, 1, result.fp, 1)
  else:
    result = m.clone()
  scal(m.len, k, result.fp, 1)

template `*`*[A: SomeFloat](k: A, v: Vector[A] or Matrix[A]): auto = v * k

template `/`*[A: SomeFloat](v: Vector[A] or Matrix[A], k: A): auto = v * (1 / k)

template `/=`*[A: SomeFloat](v: var Vector[A] or var Matrix[A], k: A) =
  v *= (1 / k)

proc `+=`*[A: SomeFloat](a: var Matrix[A], b: Matrix[A]) {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  if a.isFull and b.isFull and a.order == b.order:
    axpy(a.M * a.N, 1, b.fp, 1, a.fp, 1)
  else:
    let ap = cast[CPointer[A]](a.fp)
    if a.order == colMajor:
      for t, x in b:
        let (i, j) = t
        elColMajor(ap, a, i, j) += x
    else:
      for t, x in b:
        let (i, j) = t
        elRowMajor(ap, a, i, j) += x

proc `+`*[A: SomeFloat](a, b: Matrix[A]): Matrix[A] {. inline .} =
  if a.isFull:
    result.initLike(a)
    copy(a.len, a.fp, 1, result.fp, 1)
  else:
    result = a.clone()
  result += b

proc `-=`*[A: SomeFloat](a: var Matrix[A], b: Matrix[A]) {. inline .} =
  checkDim(a.M == b.M and a.N == a.N)
  if a.isFull and b.isFull and a.order == b.order:
    axpy(a.M * a.N, -1, b.fp, 1, a.fp, 1)
  else:
    let ap = cast[CPointer[A]](a.fp)
    if a.order == colMajor:
      for t, x in b:
        let (i, j) = t
        elColMajor(ap, a, i, j) -= x
    else:
      for t, x in b:
        let (i, j) = t
        elRowMajor(ap, a, i, j) -= x

proc `-`*[A: SomeFloat](a, b: Matrix[A]): Matrix[A] {. inline .} =
  if a.isFull:
    result.initLike(a)
    copy(a.len, a.fp, 1, result.fp, 1)
  else:
    result = a.clone()
  result -= b

proc l_2*[A: SomeFloat](m: Matrix[A]): A {. inline .} =
  if m.isFull:
    result = nrm2(m.len, m.fp, 1)
  else:
    result = sqrt(result)
    for x in m:
      result += x * x

proc l_1*[A: SomeFloat](m: Matrix[A]): A {. inline .} =
  if m.isFull:
    result = asum(m.len, m.fp, 1)
  else:
    for x in m:
      result += x.abs

proc max*[A](m: Matrix[A]): A =
  var first = true
  for x in m:
    if x > result or first:
      result = x
    first = false

proc min*[A](m: Matrix[A]): A =
  var first = true
  for x in m:
    if x < result or first:
      result = x
    first = false

proc T*[A](m: Matrix[A]): Matrix[A] =
  let mp = cast[CPointer[A]](m.fp)
  if m.order == colMajor:
    result = makeMatrixIJ(A, m.N, m.M, elColMajor(mp, m, j, i), colMajor)
  else:
    result = makeMatrixIJ(A, m.N, m.M, elRowMajor(mp, m, j, i), rowMajor)

# BLAS level 2 operations

proc `*`*[A: SomeFloat](a: Matrix[A], v: Vector[A]): Vector[A]  {. inline .} =
  checkDim(a.N == v.len)
  result = vector(newSeq[A](a.M))
  gemv(a.order, noTranspose, a.M, a.N, 1, a.fp, a.ld, v.fp, v.step, 0, result.fp, result.step)

# BLAS level 3 operations

proc `*`*[A: SomeFloat](a, b: Matrix[A]): Matrix[A] {. inline .} =
  let
    M = a.M
    K = a.N
    N = b.N
  checkDim(b.M == K)
  result = matrix[A](a.order, M, N, newSeq[A](M * N))
  if a.order == colMajor and b.order == colMajor:
    gemm(colMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, a.ld, b.fp, b.ld, 0, result.fp, result.ld)
  elif a.order == rowMajor and b.order == rowMajor:
    gemm(rowMajor, noTranspose, noTranspose, M, N, K, 1, a.fp, a.ld, b.fp, b.ld, 0, result.fp, result.ld)
  elif a.order == colMajor and b.order == rowMajor:
    gemm(colMajor, noTranspose, transpose, M, N, K, 1, a.fp, a.ld, b.fp, b.ld, 0, result.fp, result.ld)
  else:
    result.order = colMajor
    result.ld = M
    gemm(colMajor, transpose, noTranspose, M, N, K, 1, a.fp, a.ld, b.fp, b.ld, 0, result.fp, result.ld)

# Comparison

template compareApprox(a, b: Vector or Matrix): bool =
  const epsilon = 0.000001
  let
    aNorm = l_1(a)
    bNorm = l_1(b)
    dNorm = l_1(a - b)
  (dNorm / (aNorm + bNorm)) < epsilon

proc `=~`*[A: SomeFloat](v, w: Vector[A]): bool =
  compareApprox(v, w)

proc `=~`*[A: SomeFloat](v, w: Matrix[A]): bool =
  compareApprox(v, w)

template `!=~`*(a, b: Vector or Matrix): bool =
  not (a =~ b)

# Hadamard (component-wise) product
proc `|*|`*[A](a, b: Vector[A]): Vector[A] =
  checkDim(a.len == b.len)
  result = vector(newSeq[A](a.len))
  for i in 0 ..< a.len:
    result.data[i] = a[i] * b[i]

proc `|*|`*[A](a, b: Matrix[A]): Matrix[A] =
  checkDim(a.dim == b.dim)
  result.initLike(a)
  if a.isFull and b.isFull and a.order == b.order:
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

  proc fname*[A: SomeFloat](v: Vector[A]): Vector[A] =
    result = vector(newSeq[A](v.len))
    for i, x in v:
      result.data[i] = fname(x)

  proc fname*[A: SomeFloat](m: Matrix[A]): Matrix[A] =
    m.map(fname)

  export fname


template makeUniversalLocal*(fname: untyped) =
  when not compiles(fname(0'f32)):
    proc fname(x: float32): float32 = fname(x.float64).float32

  proc fname[A: SomeFloat](v: Vector[A]): Vector[A] =
    result = vector(newSeq[A](v.len))
    for i, x in v:
      result.data[i] = fname(x)

  proc fname[A: SomeFloat](m: Matrix[A]): Matrix[A] =
    m.map(fname)

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

proc mean*[A: SomeFloat](v: Vector[A]): A {.inline.} =
  sum(v) / A(v.len)

proc variance*[A: SomeFloat](v: Vector[A]): A =
  let m = v.mean
  result = v[0] - v[0]
  for x in v:
    let y = x - m
    result += y * y
  result /= A(v.len)

template stddev*[A: SomeFloat](v: Vector[A]): A =
  sqrt(variance(v))

template sum*(m: Matrix): auto = m.asVector.sum

template mean*(m: Matrix): auto = m.asVector.mean

template variance*(m: Matrix): auto = m.asVector.variance

template stddev*(m: Matrix): auto = m.asVector.stddev

# Rewrites

when defined(neoCountRewrites):
  var numRewrites = 0

  proc getRewriteCount*(): int = numRewrites

  proc resetRewriteCount*() =
    numRewrites = 0

template countRw() =
  when defined(neoCountRewrites):
    inc numRewrites

proc linearCombination[A: SomeFloat](a: A, v, w: Vector[A]): Vector[A]  {. inline .} =
  checkDim(v.len == w.len)
  result = vector(newSeq[A](v.len))
  copy(v.len, v.fp, v.step, result.fp, result.step)
  axpy(v.len, a, w.fp, w.step, result.fp, result.step)

proc linearCombinationMut[A: SomeFloat](a: A, v: var Vector[A], w: Vector[A])  {. inline .} =
  checkDim(v.len == w.len)
  axpy(v.len, a, w.fp, w.step, v.fp, v.step)

template rewriteLinearCombination*{v + `*`(w, a)}[A: SomeFloat](a: A, v, w: Vector[A]): auto =
  countRw()
  linearCombination(a, v, w)

template rewriteLinearCombinationMut*{v += `*`(w, a)}[A: SomeFloat](a: A, v: var Vector[A], w: Vector[A]): auto =
  countRw()
  linearCombinationMut(a, v, w)

# LAPACK overloads

overload(gesv, sgesv, dgesv)
overload(gebal, sgebal, dgebal)
overload(gehrd, sgehrd, dgehrd)
overload(orghr, sorghr, dorghr)
overload(hseqr, shseqr, dhseqr)

# Solvers

template solveMatrix(M, N, a, b: untyped): auto =
  # TODO: remove this requirement
  assert(a.order == colMajor, "`solve` requires a column-major matrix")
  assert(b.order == colMajor, "`solve` requires a column-major matrix")

  var
    ipvt = newSeq[int32](M)
    info: cint
    m = M.cint
    n = N.cint
    lda = a.ld.cint
    ldb = b.ld.cint
  fortran(gesv, m, n, a, lda, ipvt, b, ldb, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Left hand matrix is singular or factorization failed")

template solveVector(M, a, b: untyped): auto =
  # TODO: remove this requirement
  assert(a.order == colMajor, "`solve` requires a column-major matrix")
  var
    ipvt = newSeq[int32](M)
    info: cint
    m = M.cint
    n = 1.cint
    lda = a.ld.cint
  fortran(gesv, m, n, a, lda, ipvt, b, lda, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Left hand matrix is singular or factorization failed")

proc solve*[A: SomeFloat](a, b: Matrix[A]): Matrix[A] {.inline.} =
  checkDim(a.M == a.N, "Need a square matrix to solve the system")
  checkDim(a.M == b.M, "The dimensions are incompatible")
  var acopy = a.clone
  if b.isFull:
    result = zeros(b.M, b.N, A, b.order)
    copy(b.M * b.N, b.fp, 1, result.fp, 1)
  else:
    result = b.clone()
  solveMatrix(b.M, b.N, acopy, result)

proc solve*[A: SomeFloat](a: Matrix[A], b: Vector[A]): Vector[A] {.inline.} =
  checkDim(a.M == a.N, "Need a square matrix to solve the system")
  checkDim(a.M == b.len, "The dimensions are incompatible")
  var acopy = a.clone
  result = zeros(b.len, A)
  copy(b.len, b.fp, b.step, result.fp, result.step)
  solveVector(a.M, acopy, result)

template `\`*(a: Matrix, b: Matrix or Vector): auto = solve(a, b)

proc inv*[A: SomeFloat](a: Matrix[A]): Matrix[A] {.inline.} =
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

proc balance*[A: SomeFloat](a: Matrix[A], op = BalanceOp.Both): BalanceResult[A] =
  checkDim(a.M == a.N, "`balance` requires a square matrix")
  assert(a.order == colMajor, "`balance` requires a column-major matrix")
  result = BalanceResult[A](
    matrix: a.clone(),
    scale: newSeq[A](a.N)
  )

  var
    job = ch(op)
    n = a.N.cint
    lda = a.ld.cint
    info: cint
  # gebal(addr job, addr n, result.matrix.fp, addr n, addr result.ilo, addr result.ihi, result.scale.first, addr info)
  fortran(gebal, job, n, result.matrix, lda, result.ilo, result.ihi, result.scale, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to balance matrix")

proc hessenberg*[A: SomeFloat](a: Matrix[A]): Matrix[A] =
  checkDim(a.M == a.N, "`hessenberg` requires a square matrix")
  assert(a.order == colMajor, "`hessenberg` requires a column-major matrix")
  result = a.clone()
  var
    n = a.N.cint
    ldr = result.ld.cint
    ilo = 1.cint
    ihi = a.N.cint
    info: cint
    tau = newSeq[A](a.N)
    workSize = (-1).cint
    work = newSeq[A](1)
  # First, we call gehrd to compute the optimal work size
  # gehrd(addr n, addr ilo, addr ihi, result.fp, addr n, tau.first, work.first, addr workSize, addr info)
  fortran(gehrd, n, ilo, ihi, result, ldr, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to reduce matrix to upper Hessenberg form")
  # Then, we allocate suitable space and call gehrd again
  workSize = work[0].cint
  work = newSeq[A](workSize)
  fortran(gehrd, n, ilo, ihi, result, ldr, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to reduce matrix to upper Hessenberg form")

proc eigenvalues*[A: SomeFloat](a: Matrix[A]): EigenValues[A] =
  checkDim(a.M == a.N, "`eigenvalues` requires a square matrix")
  assert(a.order == colMajor, "`eigenvalues` requires a column-major matrix")
  var
    h = a.clone()
    n = a.N.cint
    ldh = h.ld.cint
    ilo = 1.cint
    ihi = a.N.cint
    info: cint
    tau = newSeq[A](a.N)
    workSize = (-1).cint
    work = newSeq[A](1)
  # First, we call gehrd to compute the optimal work size
  fortran(gehrd, n, ilo, ihi, h, ldh, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find eigenvalues")
  # Then, we allocate suitable space and call gehrd again
  workSize = work[0].cint
  work = newSeq[A](workSize)
  fortran(gehrd, n, ilo, ihi, h, ldh, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find eigenvalues")
  # Next, we need to find the matrix Q that transforms A into H
  var
    q = h.clone()
    ldq = q.ld.cint
  fortran(orghr, n, ilo, ihi, q, ldq, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find eigenvalues")
  var
    job = ch(EigenMode.Eigenvalues)
    compz = ch(SchurCompute.Provided)
  result = EigenValues[A](
    real: newSeq[A](a.N),
    img: newSeq[A](a.N)
  )
  fortran(hseqr, job, compz, n, ilo, ihi, h, ldh, result.real, result.img, q, ldq, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find eigenvalues")

proc schur*[A: SomeFloat](a: Matrix[A]): SchurResult[A] =
  checkDim(a.M == a.N, "`schur` requires a square matrix")
  assert(a.order == colMajor, "`schur` requires a column-major matrix")
  result.factorization = a.clone()
  var
    n = a.N.cint
    ldr = result.factorization.ld.cint
    ilo = 1.cint
    ihi = a.N.cint
    info: cint
    tau = newSeq[A](a.N)
    workSize = (-1).cint
    work = newSeq[A](1)
  # First, we call gehrd to compute the optimal work size
  fortran(gehrd, n, ilo, ihi, result.factorization, ldr, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find the Schur decomposition")
  # Then, we allocate suitable space and call gehrd again
  workSize = work[0].cint
  work = newSeq[A](workSize)
  fortran(gehrd, n, ilo, ihi, result.factorization, ldr, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find the Schur decomposition")
  # Next, we need to find the matrix Q that transforms A into H
  var
    q = result.factorization.clone()
    ldq = q.ld.cint
  fortran(orghr, n, ilo, ihi, q, ldq, tau, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find the Schur decomposition")
  var
    job = ch(EigenMode.Eigenvalues)
    compz = ch(SchurCompute.Provided)
  result.eigenvalues = EigenValues[A](
    real: newSeq[A](a.N),
    img: newSeq[A](a.N)
  )
  fortran(hseqr, job, compz, n, ilo, ihi, result.factorization, ldr,
    result.eigenvalues.real, result.eigenvalues.img, q, ldq, work, workSize, info)
  if info > 0:
    raise newException(LinearAlgebraError, "Failed to find the Schur decomposition")

# Trace and determinant

proc tr*[A](a: Matrix[A]): A =
  checkDim(a.M == a.N, "`tr` requires a square matrix")
  let ap = cast[CPointer[A]](a.fp)
  for i in 0 ..< a.M:
    result += ap[i * (1 + a.ld)]

# TODO: pick a faster decomposition
proc det*[A: SomeFloat](a: Matrix[A]): A =
  checkDim(a.M == a.N, "`det` requires a square matrix")
  result  = A(1)
  let
    s = schur(a)
    u = s.factorization
  let up = cast[CPointer[A]](u.fp)
  for i in 0 ..< a.M:
    result *= up[i * (1 + u.ld)]