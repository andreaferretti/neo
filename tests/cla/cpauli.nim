# complex linear algebra solvers

import unittest, complex, neo/cla, neo/dense

proc run() =
  suite "pauli matrix complex computations":
    test "pauli matrix":
      const
        n = 2
        cz = im(0.0)
        ci = im(1.0)
        cmi = im(-1.0)
        co = complex(1.0, 0.0)
        cmo = complex(-1.0, 0.0)
      let
        cMatI = eye(n).toComplex
        cMatmI = matrix(@[@[cmo, cz], @[cz, cmo]])

      let
        m22zr = zeros(n, n) # zeros real matrix
        si = matrix(@[@[0.0, 1.0], @[1.0, 0.0]]).toComplex(m22zr)
        sj = m22zr.toComplex(matrix(@[@[0.0, -1.0], @[1.0, 0.0]]))
        sk = matrix(@[@[1.0, 0.0], @[0.0, -1.0]]).toComplex(m22zr)
        qi = complex(0.0, -1.0) * si
        qj = complex(0.0, -1.0) * sj
        qk = complex(0.0, -1.0) * sk

      test "pauli matrix (float64)":
        check(cMatmI == cMatI * cmo)

        check(si == matrix(@[@[cz, co], @[co, cz]]))
        check(sj == matrix(@[@[cz, cmi], @[ci, cz]]))
        check(sk == matrix(@[@[co, cz], @[cz, cmo]]))

        check(qi == matrix(@[@[cz, cmi], @[cmi, cz]])) # == qI
        check(qj == matrix(@[@[cz, cmo], @[co, cz]])) # == qJ
        check(qk == matrix(@[@[cmi, cz], @[cz, ci]])) # == qK
        check(qi * qi == cMatmI) # == -I
        check(qj * qj == cMatmI) # == -I
        check(qk * qk == cMatmI) # == -I
        check(qi * qj == qk)
        check(qj * qk == qi)
        check(qk * qi == qj)
        check(qj * qi == qk * cmo)
        check(qk * qj == qi * cmo)
        check(qi * qk == qj * cmo)

      test "pauli matrix (float32)":
        check(cMatmI.to32 == (cMatI * cmo).to32)

        let
          si32 = si.to32
          sj32 = sj.to32
          sk32 = sk.to32
          qi32 = complex(0.0'f32, -1.0'f32) * si32
          qj32 = complex(0.0'f32, -1.0'f32) * sj32
          qk32 = complex(0.0'f32, -1.0'f32) * sk32

        check(si32 == matrix(@[@[cz, co], @[co, cz]]).to32)
        check(sj32 == matrix(@[@[cz, cmi], @[ci, cz]]).to32)
        check(sk32 == matrix(@[@[co, cz], @[cz, cmo]]).to32)

        check(qi32 == matrix(@[@[cz, cmi], @[cmi, cz]]).to32) # == qI
        check(qj32 == matrix(@[@[cz, cmo], @[co, cz]]).to32) # == qJ
        check(qk32 == matrix(@[@[cmi, cz], @[cz, ci]]).to32) # == qK
        check(qi32 * qi32 == cMatmI.to32) # == -I
        check(qj32 * qj32 == cMatmI.to32) # == -I
        check(qk32 * qk32 == cMatmI.to32) # == -I
        check(qi32 * qj32 == qk.to32)
        check(qj32 * qk32 == qi.to32)
        check(qk32 * qi32 == qj.to32)
        check(qj32 * qi32 == (qk * cmo).to32)
        check(qk32 * qj32 == (qi * cmo).to32)
        check(qi32 * qk32 == (qj * cmo).to32)

run()
