# complex linear algebra solvers

import unittest, complex, neo/clasolve, neo/cla, neo/dense

proc run() =
  suite "solve complex computations":
    let
      a = matrix(@[
        @[1.0, 4.0],
        @[2.0, 3.0]
      ]).toComplex
      b = matrix(@[
        @[0.0, -1.0],
        @[1.0, 0.0]
      ]).toComplex

    test "solve of a complex matrix":
      let
        ma = eye(a.N) * complex(-1.0, 0.0) # (eye(a.N) * -1.0).toComplex
        sma = matrix(@[
          @[0.6, -0.8],
          @[-0.4, 0.2]
        ]).toComplex
        mb = (eye(b.N) * -1.0).toComplex
        smb = matrix(@[
          @[0.0, -1.0],
          @[1.0, 0.0]
        ]).toComplex

      test "solve of a complex matrix (float64)":
        check(ma =~ mb)
        check(a.solve(ma) =~ sma)
        check(b.solve(mb) =~ smb)

      test "solve of a complex matrix (float32)":
        check(ma.to32 =~ mb.to32)
        check(a.to32.solve(ma.to32) =~ sma.to32)
        check(b.to32.solve(mb.to32) =~ smb.to32)

    test "solve of a complex vector":
      let
        va = (ones(a.N) * -5.0).toComplex
        sva = vector(@[-1.0, -1.0]).toComplex
        vb = (ones(b.N) * -5.0).toComplex
        svb = vector(@[-5.0, 5.0]).toComplex

      test "solve of a complex vector (float64)":
        check(a.solve(va) =~ sva)
        check(b.solve(vb) =~ svb)

      test "solve of a complex vector (float32)":
        check(a.to32.solve(va.to32) =~ sva.to32)
        check(b.to32.solve(vb.to32) =~ svb.to32)

run()
