# complex linear algebra solvers

import unittest, complex, neo/clasolve, neo/cla, neo/dense
import math
import strformat, strutils

proc fmtM*[T](a: Matrix[Complex[T]]): string=
  var r = @["[\n"]
  for i in 0..<a.M:
    r.add("[")
    for j in 0..<a.N: r.add(fmt" ({a[i, j].re:>21.17f}, {a[i, j].im:>21.17f})")
    r.add("]\n")
  r.add("]")
  result = r.join

proc fmtM*[T: SomeFloat](a: Matrix[T]): string=
  var r = @["[\n"]
  for i in 0..<a.M:
    r.add("[")
    for j in 0..<a.N: r.add(fmt" {a[i, j]:>21.17f}")
    r.add("]\n")
  r.add("]")
  result = r.join

proc checkEvsEvecs*[T](a: Matrix[Complex[T]],
  aev: seq[complex.Complex[T]], aevec: seq[Vector[Complex[T]]]): bool=
  let (evs, evecs) = a.geneig
  for j in 0..<a.N:
    check(evs.real[j].toComplex(evs.img[j]) =~ aev[j])
    check(evecs.real[j].toComplex(evecs.img[j]) =~ aevec[j])
  result = true

proc checkLambdas*[T](a: Matrix[Complex[T]],
  aev: seq[complex.Complex[T]], aevec: seq[Vector[Complex[T]]]): bool=
  let (_, S, Vh) = a.usvd # (U, S, Vh)
  # TODO: verification by some calculations
  when false:
    block:
      let (U, S, Vh) = a.usvd
      echo fmt"A = {a.fmtM}"
      echo fmt"U = {U.fmtM}"
      echo fmt"S = {S.fmtM}"
      echo fmt"Vh = {Vh.fmtM}"
      echo fmt"Vh * S.vdiag.diag * Vh.inv = {(Vh * S.vdiag.diag * Vh.inv).fmtM}"
  check(Vh * S.vdiag.diag * Vh.inv =~ a)
  check(Vh.inv * a * Vh =~ S.vdiag.diag)
  result = true

proc run() =
  suite "eigenvalues eigenvectors complex computations":
    const
      r1div5 = sqrt(5.0) / 5.0
      r1div2 = sqrt(2.0) / 2.0

    let
      are = @[
        @[
          @[1.0, 4.0],
          @[2.0, 3.0]
        ],
        @[ # complex eigenvalue
          @[0.0, -1.0],
          @[1.0, 0.0]
        ],
        @[ # diagonal matrix
          @[1.0, 0.0],
          @[0.0, 2.0]
        ],
        @[ # multiple eigenvalue solutions
          @[1.0, 0.0],
          @[0.0, 1.0]
        ]
      ]

      aevs = @[
        EigenValues[float64](
          real: @[-1.0, 5.0],
          img: @[0.0, 0.0]
        ),
        EigenValues[float64]( # complex eigenvalue
          real: @[0.0, 0.0],
          img: @[1.0, -1.0]
        ),
        EigenValues[float64]( # diagonal matrix
          real: @[1.0, 2.0],
          img: @[0.0, 0.0]
        ),
        EigenValues[float64]( # multiple eigenvalue solutions
          real: @[1.0, 1.0],
          img: @[0.0, 0.0]
        )
      ]

      aevecs = @[
        EigenVectors[float64](
          real: @[vector(@[2.0 * r1div5, -r1div5]), vector(@[r1div2, r1div2])],
          img: @[vector(@[0.0, 0.0]), vector(@[0.0, 0.0])]
        ),
        EigenVectors[float64]( # complex eigenvalue
          real: @[vector(@[r1div2, 0.0]), vector(@[r1div2, 0.0])],
          img: @[vector(@[0.0, -r1div2]), vector(@[0.0, r1div2])]
        ),
        EigenVectors[float64]( # diagonal matrix
          real: @[vector(@[1.0, 0.0]), vector(@[0.0, 1.0])],
          img: @[vector(@[0.0, 0.0]), vector(@[0.0, 0.0])]
        ),
        EigenVectors[float64]( # multiple eigenvalue solutions
          real: @[vector(@[1.0, 0.0]), vector(@[0.0, 1.0])],
          img: @[vector(@[0.0, 0.0]), vector(@[0.0, 0.0])]
        )
      ]

    test "eigenvalues eigenvectors of a complex matrix (float64)":
      for i in 0..<are.len:
        let a = matrix(are[i]).toComplex
        var aev = newSeq[complex.Complex[float64]](a.N)
        var aevec = newSeq[Vector[Complex[float64]]](a.N)
        for j in 0..<a.N:
          aev[j] = aevs[i].real[j].toComplex(aevs[i].img[j])
          aevec[j] = aevecs[i].real[j].toComplex(aevecs[i].img[j])
        test fmt"checkEvsEvecs(f64) {i}":
          check(checkEvsEvecs(a, aev, aevec))
        test fmt"checkLambdas(f64) {i}":
          check(checkLambdas(a, aev, aevec))

    test "eigenvalues eigenvectors of a complex matrix (float32)":
      for i in 0..<are.len:
        let a32 = matrix(are[i]).toComplex.to32
        var aev32 = newSeq[complex.Complex[float32]](a32.N)
        var aevec32 = newSeq[Vector[Complex[float32]]](a32.N)
        for j in 0..<a32.N:
          aev32[j] = aevs[i].real[j].toComplex(aevs[i].img[j]).to32
          aevec32[j] = aevecs[i].real[j].toComplex(aevecs[i].img[j]).to32
        test fmt"checkEvsEvecs(f32) {i}":
          check(checkEvsEvecs(a32, aev32, aevec32))
        test fmt"checkLambdas(f32) {i}":
          check(checkLambdas(a32, aev32, aevec32))

run()
