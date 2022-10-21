# complex linear algebra solvers

import unittest, complex, neo/clasolve, neo/cla, neo/dense

proc run() =
  suite "determinant complex computations":
    test "determinant of a complex matrix":
      let
        a = matrix(@[
          @[1.0, 4.0],
          @[2.0, 3.0]
        ]).toComplex
        b = matrix(@[
          @[0.0, -1.0],
          @[1.0, 0.0]
        ]).toComplex
        m = matrix(@[
          @[1.0, 0.0, 1.0],
          @[0.0, 1.0, 2.0]
        ]).toComplex

      test "determinant of a complex matrix (float64)":
        check((det(a) + complex(5.0, 0.0)) =~ im(0.0)) # 5 * -1
        check((det(b) + complex(-1.0, 0.0)) =~ im(0.0)) # i * -i
        check(det(m) =~ im(0.0)) # TODO: must be assertion error ?

      test "determinant of a complex matrix (float32)":
        check((a.to32.det + complex(5.0'f32, 0.0'f32)) =~ im(0.0'f32)) # 5 * -1
        check((b.to32.det + complex(-1.0'f32, 0.0'f32)) =~ im(0.0'f32)) # i * -i
        check(m.to32.det =~ im(0.0'f32)) # TODO: must be assertion error ?

run()
