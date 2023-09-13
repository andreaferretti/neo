# complex linear algebra
#
# see also clasolve.nim
#

import nimblas
import sequtils, complex
import sugar
import ./dense

template czelColMajor[T](ap: CPointer[T]; a, i, j: untyped): untyped=
  (cast[ptr Complex[T]](ap[2 * (j * a.ld + i)].addr))[] # complex size

template czelRowMajor[T](ap: CPointer[T]; a, i, j: untyped): untyped=
  (cast[ptr Complex[T]](ap[2 * (i * a.ld + j)].addr))[] # complex size

proc all*[T](v: Vector[Complex[T]];
  pred: proc(i: int; x: complex.Complex[T]): bool {.closure.}): bool=
  result = true
  for i, e in v:
    if not pred(i, e):
      result = false
      break

proc all*[T](a: Matrix[Complex[T]];
  pred: proc(i, j: int; x: complex.Complex[T]): bool {.closure.}): bool=
  result = true
  for ij, e in a:
    if not pred(ij[0], ij[1], e):
      result = false
      break

template tr*[T](a: Matrix[T]): T=
  a.trace

template tr*[T](a: Matrix[Complex[T]]): complex.Complex[T]=
  a.trace

template trace*[T](a: Matrix[T]): T=
  a.vdiag.sum

template trace*[T](a: Matrix[Complex[T]]): complex.Complex[T]=
  a.vdiag.sum

proc vdiag*[T](a: Matrix[T]): Vector[T] {.noInit.}=
  let k = min(a.M, a.N)
  result = zeros(k, T)
  for i in 0..<k: result[i] = a[i, i]

proc vdiag*[T](a: Matrix[Complex[T]]): Vector[Complex[T]] {.noInit.}=
  let k = min(a.M, a.N)
  result = constantVector(k, complex.complex[T](0.0, 0.0))
  for i in 0..<k: result[i] = a[i, i]

proc diag*[T](v: Vector[T], order=colMajor): Matrix[T] {.noInit.}=
  result = zeros(v.len, v.len, T, order=order)
  result.shape.incl(Diagonal)
  for i in 0..<v.len: result[i, i] = v[i]

proc diag*[T](v: Vector[Complex[T]], order=colMajor): Matrix[Complex[T]] {.noInit.}=
  result = constantMatrix(v.len, v.len, complex.complex[T](0.0, 0.0), order=order)
  result.shape.incl(Diagonal)
  for i in 0..<v.len: result[i, i] = v[i]

proc realize*[T](c: complex.Complex[T]):
  tuple[real: T, img: T]=
  result = (c.re, c.im)

proc realize*[T](v: Vector[Complex[T]]):
  tuple[real: Vector[T], img: Vector[T]]=
  result = (v.map(x => x.re), v.map(x => x.im))

proc realize*[T](a: Matrix[Complex[T]]):
  tuple[real: Matrix[T], img: Matrix[T]]=
  result = (a.map(x => x.re), a.map(x => x.im))

proc toComplex*[T](re: T, im: T): complex.Complex[T]=
  result = complex.complex[T](re, im)

proc toComplex*[T](re: T): complex.Complex[T]=
  result = complex.complex[T](re, 0.0)

proc toComplex*[T](re: Vector[T], im: Vector[T]): Vector[Complex[T]]=
  result = makeVector(re.len,
    proc(i: int): Complex[T]= complex.complex[T](re[i], im[i]))

proc toComplex*[T](re: Vector[T]): Vector[Complex[T]]=
  result = re.toComplex(zeros(re.len, T))

proc toComplex*[T](re: Matrix[T], im: Matrix[T]): Matrix[Complex[T]]=
  result = makeMatrix(re.M, re.N,
    proc(i, j: int): Complex[T]= complex.complex[T](re[i, j], im[i, j]),
    order=re.order)

proc toComplex*[T](re: Matrix[T]): Matrix[Complex[T]]=
  result = re.toComplex(zeros(re.M, re.N, T, order=re.order))

proc to32*(c: complex.Complex[float64]): complex.Complex[float32]=
  let (re, im) = c.realize
  result = toComplex[float32](re, im) # not use re.toComplex[float32](im)

proc to64*(c: complex.Complex[float32]): complex.Complex[float64]=
  let (re, im) = c.realize
  result = toComplex[float64](re, im) # not use re.toComplex[float64](im)

proc to32*(v: Vector[Complex[float64]]): Vector[Complex[float32]]=
  let (re, im) = v.realize
  result = re.to32.toComplex(im.to32)

proc to64*(v: Vector[Complex[float32]]): Vector[Complex[float64]]=
  let (re, im) = v.realize
  result = re.to64.toComplex(im.to64)

proc to32*(a: Matrix[Complex[float64]]): Matrix[Complex[float32]]=
  let (re, im) = a.realize
  result = re.to32.toComplex(im.to32)

proc to64*(a: Matrix[Complex[float32]]): Matrix[Complex[float64]]=
  let (re, im) = a.realize
  result = re.to64.toComplex(im.to64)

proc conjugate*[T](v: Vector[Complex[T]]): Vector[Complex[T]]=
  result = v.map(x => x.conjugate)

proc conjugate*[T](a: Matrix[Complex[T]]): Matrix[Complex[T]]=
  result = a.map(x => x.conjugate)

proc H*[T](a: Matrix[Complex[T]]): Matrix[Complex[T]]=
  result = a.T.conjugate # or a.conjugate.T

template compareApprox[T](x, y: complex.Complex[T]): bool=
  # TODO: more fast
  const epsilon = 0.000001
  const czepsilonabs2 = complex.complex[T](epsilon, epsilon).abs2
  (x - y).abs2 < czepsilonabs2

proc `=~`*[T](x, y: complex.Complex[T]): bool=
  result = compareApprox(x, y)

template `!=~`*[T](x, y: complex.Complex[T]): bool=
  not (x =~ y)

proc `=~`*[T](v, w: Vector[Complex[T]]): bool=
  # TODO: more fast
  result = v.all(proc(i: int; x: complex.Complex[T]): bool= x =~ w[i])

template `!=~`*[T](v, w: Vector[Complex[T]]): bool=
  not (v =~ w)

proc `=~`*[T](a, b: Matrix[Complex[T]]): bool=
  # TODO: more fast
  result = a.all(proc(i, j: int; x: complex.Complex[T]): bool= x =~ b[i, j])

template `!=~`*[T](a, b: Matrix[Complex[T]]): bool=
  not (a =~ b)

proc `+=`*[T](v: var Vector[Complex[T]]; w: Vector[Complex[T]])=
  assert v.len == w.len
  var
    k = v.len
    alpha = complex.complex[T](1.0, 0.0) # v := w + v
    vw = w # needless .clone
    incx = 1
    incy = 1
  axpy(k, alpha.addr, vw.fp, incx, v.fp, incy)

proc `+`*[T](v, w: Vector[Complex[T]]): Vector[Complex[T]]=
  result = v.clone
  result += w

proc `+=`*[T](a: var  Matrix[Complex[T]]; b: Matrix[Complex[T]])=
  assert a.M == b.M and a.N == b.N # plus for same shape Matrix
  if a.isFull and b.isFull and a.order == b.order:
    var
      k = a.M * a.N
      alpha = complex.complex[T](1.0, 0.0) # a := b + a
      vw = b # needless .clone
      incx = 1
      incy = 1
    axpy(k, alpha.addr, vw.fp, incx, a.fp, incy)
  else:
    let ap = cast[CPointer[T]](a.fp) # complex size
    if a.order == colMajor:
      for t, x in b:
        let (i, j) = t
        czelColMajor(ap, a, i, j) += x
    else:
      for t, x in b:
        let (i, j) = t
        czelRowMajor(ap, a, i, j) += x

proc `+`*[T](a, b: Matrix[Complex[T]]): Matrix[Complex[T]]=
  result = a.clone
  result += b

proc `-=`*[T](v: var Vector[Complex[T]]; w: Vector[Complex[T]])=
  assert v.len == w.len
  var
    k = v.len
    alpha = complex.complex[T](-1.0, 0.0) # v := -w + v
    vw = w # needless .clone
    incx = 1
    incy = 1
  axpy(k, alpha.addr, vw.fp, incx, v.fp, incy)

proc `-`*[T](v, w: Vector[Complex[T]]): Vector[Complex[T]]=
  result = v.clone
  result -= w

proc `-=`*[T](a: var  Matrix[Complex[T]]; b: Matrix[Complex[T]])=
  assert a.M == b.M and a.N == b.N # plus for same shape Matrix
  if a.isFull and b.isFull and a.order == b.order:
    var
      k = a.M * a.N
      alpha = complex.complex[T](-1.0, 0.0) # a := -b + a
      vw = b # needless .clone
      incx = 1
      incy = 1
    axpy(k, alpha.addr, vw.fp, incx, a.fp, incy)
  else:
    let ap = cast[CPointer[T]](a.fp) # complex size
    if a.order == colMajor:
      for t, x in b:
        let (i, j) = t
        czelColMajor(ap, a, i, j) -= x
    else:
      for t, x in b:
        let (i, j) = t
        czelRowMajor(ap, a, i, j) -= x

proc `-`*[T](a, b: Matrix[Complex[T]]): Matrix[Complex[T]]=
  result = a.clone
  result -= b

proc `*`*[T](v: Vector[T]; c: complex.Complex[T]): Vector[Complex[T]]=
  result = v.toComplex
  result *= c

proc `*=`*[T](v: var Vector[Complex[T]]; c: complex.Complex[T])=
  var
    k = v.len
    alpha = c # v := cv
    incx = 1
  scal(k, alpha.addr, v.fp, incx)

proc `*`*[T](v: Vector[Complex[T]]; c: complex.Complex[T]):
  Vector[Complex[T]]=
  result = v.clone
  result *= c

proc `*`*[T](v, w: Vector[Complex[T]]): complex.Complex[T]=
  result = v.dotu(w)

proc dotu*[T](v, w: Vector[Complex[T]]): complex.Complex[T]=
  assert v.len == w.len
  var
    k = v.len
    vv = v # needless .clone
    vw = w # needless .clone
    incx = 1
    incy = 1
  result = complex.complex[T](0.0, 0.0)
  dotu(k, vv.fp, incx, vw.fp, incy, result.addr)

proc dotc*[T](v, w: Vector[Complex[T]]): complex.Complex[T]=
  # v.dotc(w) == v.conjugate.dotu(w)
  assert v.len == w.len
  var
    k = v.len
    vv = v # needless .clone
    vw = w # needless .clone
    incx = 1
    incy = 1
  result = complex.complex[T](0.0, 0.0)
  dotc(k, vv.fp, incx, vw.fp, incy, result.addr)

proc `*`*[T](a: Matrix[T]; c: complex.Complex[T]): Matrix[Complex[T]]=
  result = a.toComplex
  result *= c

proc `*=`*[T](a: var Matrix[Complex[T]]; c: complex.Complex[T])=
  var
    k = a.M * a.N
    alpha = c # a := ca
    incx = 1
  scal(k, alpha.addr, a.fp, incx)

proc `*`*[T](a: Matrix[Complex[T]]; c: complex.Complex[T]):
  Matrix[Complex[T]]=
  result = a.clone
  result *= c

proc `*=`*[T](a: var Matrix[Complex[T]]; b: Matrix[Complex[T]])=
  a = a * b

proc `*`*[T](a, b: Matrix[Complex[T]]): Matrix[Complex[T]]=
  assert a.N == b.M
  var
    m = a.M
    n = b.N
    k = b.M # == a.N
    alpha = complex.complex[T](1.0, 0.0)
    va = a # needless .clone
    vb = b # needless .clone
    beta = complex.complex[T](-1.0, 0.0)
  result = constantMatrix(m, n, complex.complex[T](0.0, 0.0), order=a.order)
  if a.order == colMajor and b.order == colMajor:
    gemm(result.order, noTranspose, noTranspose, m, n, k,
      alpha.addr, va.fp, va.ld, vb.fp, vb.ld,
      beta.addr, result.fp, result.ld) # result colMajor
  elif a.order == rowMajor and b.order == rowMajor:
    gemm(result.order, noTranspose, noTranspose, m, n, k,
      alpha.addr, va.fp, va.ld, vb.fp, vb.ld,
      beta.addr, result.fp, result.ld) # result rowMajor
  elif a.order == colMajor and b.order == rowMajor:
    gemm(result.order, noTranspose, transpose, m, n, k,
      alpha.addr, va.fp, va.ld, vb.fp, vb.ld,
      beta.addr, result.fp, result.ld) # result colMajor
  else: # a.order == rowMajor and b.order == colMajor
    result.order = colMajor
    result.ld = m
    gemm(result.order, transpose, noTranspose, m, n, k,
      alpha.addr, va.fp, va.ld, vb.fp, vb.ld,
      beta.addr, result.fp, result.ld) # result colMajor != a.order

template dotc*[T](a, b: Matrix[Complex[T]]): complex.Complex[T]=
  # Hilbert-Schmidt Inner Product
  result = (a.H * b).tr

proc `*`*[T](a: Matrix[Complex[T]]; v: Vector[Complex[T]]): Vector[Complex[T]]=
  assert a.N == v.len
  var
    m = a.M
    n = v.len # == a.N
    alpha = complex.complex[T](1.0, 0.0)
    va = a # needless .clone
    vv = v # needless .clone
    beta = complex.complex[T](-1.0, 0.0)
    incx = 1
    incy = 1
  result = constantVector(n, complex.complex[T](0.0, 0.0))
  gemv(a.order, noTranspose, m, n,
    alpha.addr, va.fp, m, vv.fp, incx,
    beta.addr, result.fp, incy)

template `*`*[T](c: complex.Complex[T],
  v: Vector[T] or Matrix[T]): auto= v * c

template `*`*[T](c: complex.Complex[T],
  v: Vector[Complex[T]] or Matrix[Complex[T]]): auto= v * c

template `*=`*[T](v: var Vector[Complex[T]] or var Matrix[Complex[T]]; k: T)=
  v *= k.toComplex

template `*`*[T](v: Vector[Complex[T]] or Matrix[Complex[T]]; k: T): auto=
  v * k.toComplex

template `*`*[T](k: T,
  v: Vector[Complex[T]] or Matrix[Complex[T]]): auto= v * k

template `/=`*[T](v: var Vector[T] or Matrix[T],
  c: complex.Complex[T])=
  v *= (complex.complex[T](1.0, 0.0) / c)

template `/`*[T](v: Vector[T] or Matrix[T],
  c: complex.Complex[T]): auto=
  v * (complex.complex[T](1.0, 0.0) / c)

template `/=`*[T](v: var Vector[Complex[T]] or Matrix[Complex[T]],
  c: complex.Complex[T])=
  v *= (complex.complex[T](1.0, 0.0) / c)

template `/`*[T](v: Vector[Complex[T]] or Matrix[Complex[T]],
  c: complex.Complex[T]): auto=
  v * (complex.complex[T](1.0, 0.0) / c)

template `/=`*[T](v: var Vector[Complex[T]] or var Matrix[Complex[T]]; k: T)=
  v /= k.toComplex

template `/`*[T](v: Vector[Complex[T]] or Matrix[Complex[T]]; k: T): auto=
  v / k.toComplex

template sameVector*[T](a, b: Matrix[T]): bool=
  a.asVector == b.asVector

template sameVector*[T](a, b: Matrix[Complex[T]]): bool=
  a.asVector == b.asVector

template `-`*[T](v: Vector[Complex[T]] or Matrix[Complex[T]]): auto=
  v * complex.complex[T](-1.0, 0.0)

template `-`*[T](v: Vector[T] or Matrix[T]): auto=
  v * -1.0
