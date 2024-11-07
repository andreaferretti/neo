# complex linear algebra solvers

import unittest, complex, neo/clasolve, neo/cla, neo/dense

proc run() =
  suite "invert complex computations":
    test "invert of a complex matrix":
      let
        e = matrix(@[
          @[1.0, 0.0],
          @[0.0, 1.0]
        ]).toComplex
        a = matrix(@[
          @[1.0, 4.0],
          @[2.0, 3.0]
        ]).toComplex
        ainv = matrix(@[
          @[-0.6, 0.8],
          @[0.4, -0.2]
        ]).toComplex
        b = matrix(@[
          @[0.0, -1.0],
          @[1.0, 0.0]
        ]).toComplex # same as qj
        binv = matrix(@[
          @[0.0, 1.0],
          @[-1.0, 0.0]
        ]).toComplex # same as qj.H
        qi = matrix(@[
          @[0.0, 1.0],
          @[1.0, 0.0]
        ]) * im(-1.0) # Unitary Matrix
        qj = matrix(@[
          @[im(0.0), im(-1.0)],
          @[im(1.0), im(0.0)]
        ]) * im(-1.0) # Unitary Matrix
        qk = matrix(@[
          @[1.0, 0.0],
          @[0.0, -1.0]
        ]) * im(-1.0) # Unitary Matrix

      test "invert of a complex matrix (float64)":
        check(inv(e) =~ e)
        check(inv(a) =~ ainv)
        check(inv(b) =~ binv)

        check(qi.inv =~ qi.H)
        check(qi * qi.H =~ e)

        check(qj.inv =~ qj.H)
        check(qj * qj.H =~ e)

        check(qk.inv =~ qk.H)
        check(qk * qk.H =~ e)

      test "invert of a complex matrix (float32)":
        let
          e32 = e.to32
          qi32 = qi.to32
          qj32 = qj.to32
          qk32 = qk.to32

        check(e32.inv =~ e32)
        check(a.to32.inv =~ ainv.to32)
        check(b.to32.inv =~ binv.to32)

        check(qi32.inv =~ qi32.H)
        check(qi32 * qi32.H =~ e32)

        check(qj32.inv =~ qj32.H)
        check(qj32 * qj32.H =~ e32)

        check(qk32.inv =~ qk32.H)
        check(qk32 * qk32.H =~ e32)

run()
