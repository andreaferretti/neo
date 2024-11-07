# complex linear algebra privates (re im decomposition)

import complex, ../dense, ../cla

proc ae[T](x, y: complex.Complex[T]): bool=
  result = x =~ y

proc ae[T](v, w: Vector[Complex[T]]): bool=
  result = v =~ w

proc ae_ri[T](v, w: Vector[Complex[T]]): bool=
  let (vr, vi) = v.realize
  let (wr, wi) = w.realize
  result = vr =~ wr and vi =~ wi

proc ae[T](a, b: Matrix[Complex[T]]): bool=
  result = a =~ b

proc ae_ri[T](a, b: Matrix[Complex[T]]): bool=
  let (ar, ai) = a.realize
  let (br, bi) = b.realize
  result = ar =~ br and ai =~ bi

proc plus[T](v, w: Vector[Complex[T]]): Vector[Complex[T]]=
  result = v + w

proc plus_ri[T](v, w: Vector[Complex[T]]): Vector[Complex[T]]=
  let (vr, vi) = v.realize
  let (wr, wi) = w.realize
  result = (vr + wr).toComplex(vi + wi)

proc plus[T](a, b: Matrix[Complex[T]]): Matrix[Complex[T]]=
  result = a + b

proc plus_ri[T](a, b: Matrix[Complex[T]]): Matrix[Complex[T]]=
  let (ar, ai) = a.realize
  let (br, bi) = b.realize
  result = (ar + br).toComplex(ai + bi)

proc dotu_ri[T](v, w: Vector[Complex[T]]): complex.Complex[T]=
  let (vr, vi) = v.realize
  let (wr, wi) = w.realize
  result = (vr * wr - vi * wi).toComplex(vr * wi + vi * wr)

proc dotc_ri[T](v, w: Vector[Complex[T]]): complex.Complex[T]=
  let (vr, vi) = v.realize
  let (wr, wi) = w.realize
  result = (vr * wr + vi * wi).toComplex(vr * wi - vi * wr)

proc dotu[T](a, b: Matrix[Complex[T]]): Matrix[Complex[T]]=
  result = a * b

proc dotu_ri[T](a, b: Matrix[Complex[T]]): Matrix[Complex[T]]=
  let (ar, ai) = a.realize
  let (br, bi) = b.realize
  result = (ar * br - ai * bi).toComplex(ar * bi + ai * br)

proc dotu[T](a: Matrix[Complex[T]]; v: Vector[Complex[T]]): Vector[Complex[T]]=
  result = a * v

proc dotu_ri[T](a: Matrix[Complex[T]]; v: Vector[Complex[T]]):
  Vector[Complex[T]]=
  let (ar, ai) = a.realize
  let (vr, vi) = v.realize
  result = (ar * vr - ai * vi).toComplex(ar * vi + ai * vr)
