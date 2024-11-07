# complex linear algebra solvers

import unittest, complex, neo/cla, neo/dense

proc run() =
  suite "hermitian matrix complex computations":
    test "hermitian matrix":
      let
        cm22h = matrix(@[
          @[complex(1.0, 0.0), im(1.0)],
          @[im(-1.0), complex(1.0, 0.0)]]) # [[1, i], [-i, 1]]
        cm22t = matrix(@[
          @[complex(1.0, 0.0), im(-1.0)],
          @[im(1.0), complex(1.0, 0.0)]]) # [[1, -i], [i, 1]]

      test "hermitian matrix (float64)":
        check(not (cm22h.t == cm22h))
        check(not (cm22h.t =~ cm22h))
        check(cm22h.H == cm22h)
        check(cm22h.H =~ cm22h)
        check(cm22t.H == cm22t)
        check(cm22t.H =~ cm22t)

      test "hermitian matrix (float32)":
        let
          cm22h32 = cm22h.to32
          cm22t32 = cm22t.to32

        check(not (cm22h32.t == cm22h32))
        check(not (cm22h32.t =~ cm22h32))
        check(cm22h32.H == cm22h32)
        check(cm22h32.H =~ cm22h32)
        check(cm22t32.H == cm22t32)
        check(cm22t32.H =~ cm22t32)

run()
