# complex linear algebra solvers
#
# see also cla.nim
#

import sequtils, complex
import ./dense, ./private/neocommon
import ./cla, ./private/clacommon

## use czfortran() for lapack_complex_* forked from fortran() in neocommon.nim
overload(geev, cgeev, zgeev)
overload(gesv, cgesv, zgesv)

proc usvd*[T: SomeFloat](a: Matrix[Complex[T]]):
  tuple[U: Matrix[Complex[T]], S: Matrix[Complex[T]], Vh: Matrix[Complex[T]]]=
  var
    m = a.M.int32
    n = a.N.int32
    vS = a.clone # overwritten by result (Diag or Triangular)
    vW = constantVector(n, complex.complex[T](0.0, 0.0)) # W: EigenValues
    jobvl = cstring"V"
    ldvl = m
    vVL = constantMatrix(m, m, complex.complex[T](0.0, 0.0), order=a.order)
    jobvr = cstring"V"
    ldvr = n
    vVR = constantMatrix(n, n, complex.complex[T](0.0, 0.0), order=a.order) # VR: EigenVectors
    lwork = n * 4 # or get before 2pass [c|z]geev
    vwork = constantVector(lwork, complex.complex[T](0.0, 0.0))
    rwork = zeros(lwork, T)
    info = 0'i32
  czfortran(geev, jobvl, jobvr, n, vS, m, vW,
    vVL, ldvl, vVR, ldvr, vwork, lwork, rwork, info)
  # TODO: convert vS to vS.vdiag and vVL to vVR.inv for compatible GE SVD
  assert vW == vS.vdiag
  result = (vVL, vS, vVR)

proc usvd*[T: SomeFloat](a: Matrix[T]):
  tuple[U: Matrix[Complex[T]], S: Matrix[Complex[T]], Vh: Matrix[Complex[T]]]=
  result = a.toComplex.usvd

proc geneig*[T](a: Matrix[Complex[T]]): (EigenValues[T], EigenVectors[T])=
  let (_, S, Vh) = a.usvd # (U, S, Vh)
  var (ers, eis) = (newSeq[T](a.N), newSeq[T](a.N))
  var (vrs, vis) = (newSeq[Vector[T]](a.N), newSeq[Vector[T]](a.N))
  for i in 0..<a.N:
    (ers[i], eis[i]) = S[i, i].realize
    (vrs[i], vis[i]) = Vh.column(i).realize # .T
  result = (
    EigenValues[T](real: ers, img: eis),
    EigenVectors[T](real: vrs, img: vis))

proc geneig*[A: SomeFloat](a: Matrix[A]): (EigenValues[A], EigenVectors[A])=
  result = a.toComplex.geneig

proc solve*[T: SomeFloat](a, b: Matrix[Complex[T]]): Matrix[Complex[T]]=
  var
    m = a.M.int32
    n = a.N.int32 # == b.M
    nrhs = b.N.int32
    va = a.clone
    ipiv = newSeq[int32](n)
    info = 0'i32
  result = b.clone
  czfortran(gesv, n, nrhs, va, m, ipiv, result, n, info)

proc solve*[T: SomeFloat](a: Matrix[Complex[T]]; v: Vector[Complex[T]]):
  Vector[Complex[T]]=
  result = a.solve(matrix(@[v.toSeq], order=a.order).t).column(0)

proc inv*[T](a: Matrix[Complex[T]]): Matrix[Complex[T]]=
  result = a.solve(eye(a.N, T, order=a.order).toComplex)

proc det*[T](a: Matrix[Complex[T]]): complex.Complex[T]=
  # assert a.M == a.N # TODO: must be assertion error ?
  if a.M != a.N: result = complex.complex[T](0.0, 0.0)
  else:
    let (_, S, _) = a.usvd # (U, S, Vh)
    result = complex.complex[T](1.0, 0.0)
    for i in 0..<a.N: result *= S[i, i]
