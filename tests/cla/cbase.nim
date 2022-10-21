# complex linear algebra solvers

import unittest, complex, neo/cla, neo/dense

proc run() =
  suite "basic complex computations":
    # TODO: assert
    # TODO: support and check for sparse matrix
    # TODO: support and check for cuda cudadense cudasparse

    const
      ci = im(1.0)
      cmi = im(-1.0)
      coi = complex.complex(1.0, 1.0)
      cmomi = complex.complex(-1.0, -1.0)
      ci32 = ci.to32
      cmi32 = cmi.to32
      coi32 = coi.to32
      cmomi32 = cmomi.to32

    test "Complex[float32] Complex[float64] of a complex vector matrix":
      let
        vf32re = vector(@[0.1'f32, 0.2'f32])
        vf32 = vf32re.toComplex
        mf32re = matrix(@[
          @[0.1'f32, 0.4'f32],
          @[0.2'f32, 0.3'f32]
        ])
        mf32 = mf32re.toComplex
        vf64re = vector(@[0.1'f64, 0.2'f64])
        vf64 = vf64re.toComplex
        mf64re = matrix(@[
          @[0.1'f64, 0.4'f64],
          @[0.2'f64, 0.3'f64]
        ])
        mf64 = mf64re.toComplex
        vf64delta5 = vector(@[0.10001'f64, 0.19999'f64]).toComplex
        mf64delta5 = matrix(@[
          @[0.10001'f64, 0.39999'f64],
          @[0.19999'f64, 0.30001'f64]
        ]).toComplex
        vf64delta6 = vector(@[0.100001'f64, 0.199999'f64]).toComplex
        mf64delta6 = matrix(@[
          @[0.100001'f64, 0.399999'f64],
          @[0.199999'f64, 0.300001'f64]
        ]).toComplex

      test "to32 of a complex vector matrix (float64)":
        # check(vf32 =~ vf64) # type mismatch on compile
        # check(mf32 =~ mf64) # type mismatch on compile
        check(vf32 =~ vf64.to32)
        check(mf32 =~ mf64.to32)

        check((vf64 * 2.0) == (vf64re * coi * coi.conjugate))
        check((mf64 * 2.0) == (mf64re * coi * coi.conjugate))

      test "to64 of a complex vector matrix (float32)":
        check(vf32.to64 =~ vf64)
        check(mf32.to64 =~ mf64)

        check((vf32 * 2.0'f32) == (vf32re * coi32 * coi32.conjugate))
        check((mf32 * 2.0'f32) == (mf32re * coi32 * coi32.conjugate))

      test "epsilon precision of a complex vector matrix (float64)":
        check(not (vf64delta5 == vf64))
        check(not (vf64delta5 =~ vf64)) # epsilon precision
        check(not (mf64delta5 == mf64))
        check(not (mf64delta5 =~ mf64)) # epsilon precision

        check(not (vf64delta6 == vf64))
        check(vf64delta6 =~ vf64) # epsilon precision
        check(not (mf64delta6 == mf64))
        check(mf64delta6 =~ mf64) # epsilon precision

      test "epsilon precision of a complex vector matrix (float32)":
        let
          vf32delta5 = vf64delta5.to32
          mf32delta5 = mf64delta5.to32
          vf32delta6 = vf64delta6.to32
          mf32delta6 = mf64delta6.to32

        check(not (vf32delta5 == vf32))
        check(not (vf32delta5 =~ vf32)) # epsilon precision
        check(not (mf32delta5 == mf32))
        check(not (mf32delta5 =~ mf32)) # epsilon precision

        check(not (vf32delta6 == vf32))
        check(vf32delta6 =~ vf32) # epsilon precision
        check(not (mf32delta6 == mf32))
        check(mf32delta6 =~ mf32) # epsilon precision

    test "unary operator or real times of a complex vector matrix":
      let
        v3are = vector(@[-1.0, 0.1, -0.1])
        v3a = v3are.toComplex
        v3mare = vector(@[1.0, -0.1, 0.1])
        v3ma = v3mare.toComplex
        m22are = matrix(@[@[0.0, -1.0], @[1.0, 0.0]])
        m22a = m22are.toComplex
        m22mare = matrix(@[@[0.0, 1.0], @[-1.0, 0.0]])
        m22ma = m22mare.toComplex

      test "unary operator or real times of a complex vector matrix (float64)":
        check(0.0 - coi == cmomi) # OK only for scalar
        check((-coi) == cmomi)
        check(-1.0 * coi == cmomi)

        ## type mismatch (at position: 2)
        # check(0.0 - v3are == v3mare)
        # check(0.0 - v3a == v3ma)
        # check(0.0 - m22are == m22mare)
        # check(0.0 - m22a == m22ma)

        check((-v3are) == v3mare)
        check((-v3a) == v3ma)
        check((-m22are) == m22mare)
        check((-m22a) == m22ma)

        check(-1.0 * v3are == v3mare)
        check(-1.0 * v3a == v3ma)
        check(-1.0 * m22are == m22mare)
        check(-1.0 * m22a == m22ma)

        check(v3are / -1.0 == v3mare)
        check(v3a / -1.0 == v3ma)
        check(m22are / -1.0 == m22mare)
        check(m22a / -1.0 == m22ma)

        var
          v3vre = v3are.clone
          v3v = v3a.clone
          m22vre = m22are.clone
          m22v = m22a.clone
          vd3vre = v3are.clone
          vd3v = v3a.clone
          md22vre = m22are.clone
          md22v = m22a.clone
        v3vre *= -1.0
        v3v *= -1.0
        m22vre *= -1.0
        m22v *= -1.0
        vd3vre *= -1.0
        vd3v *= -1.0
        md22vre *= -1.0
        md22v *= -1.0
        check(v3vre == v3mare)
        check(v3v == v3ma)
        check(m22vre == m22mare)
        check(m22v == m22ma)
        check(vd3vre == v3mare)
        check(vd3v == v3ma)
        check(md22vre == m22mare)
        check(md22v == m22ma)

      test "unary operator or real times of a complex vector matrix (float32)":
        let
          v3are32 = v3are.to32
          v3a32 = v3a.to32
          v3mare32 = v3mare.to32
          v3ma32 = v3ma.to32
          m22are32 = m22are.to32
          m22a32 = m22a.to32
          m22mare32 = m22mare.to32
          m22ma32 = m22ma.to32

        check(0.0'f32 - coi32 == cmomi32) # OK only for scalar
        check((-coi32) == cmomi32)
        check(-1.0'f32 * coi32 == cmomi32)

        ## type mismatch (at position: 2)
        # check(0.0'f32 - v3are32 == v3mare32)
        # check(0.0'f32 - v3a32 == v3ma32)
        # check(0.0'f32 - m22are32 == m22mare32)
        # check(0.0'f32 - m22a32 == m22ma32)

        check((-v3are32) == v3mare32)
        check((-v3a32) == v3ma32)
        check((-m22are32) == m22mare32)
        check((-m22a32) == m22ma32)

        check(-1.0'f32 * v3are32 == v3mare32)
        check(-1.0'f32 * v3a32 == v3ma32)
        check(-1.0'f32 * m22are32 == m22mare32)
        check(-1.0'f32 * m22a32 == m22ma32)

        check(v3are32 / -1.0'f32 == v3mare32)
        check(v3a32 / -1.0'f32 == v3ma32)
        check(m22are32 / -1.0'f32 == m22mare32)
        check(m22a32 / -1.0'f32 == m22ma32)

        var
          v3vre32 = v3are32.clone
          v3v32 = v3a32.clone
          m22vre32 = m22are32.clone
          m22v32 = m22a32.clone
          vd3vre32 = v3are32.clone
          vd3v32 = v3a32.clone
          md22vre32 = m22are32.clone
          md22v32 = m22a32.clone
        v3vre32 *= -1.0'f32
        v3v32 *= -1.0'f32
        m22vre32 *= -1.0'f32
        m22v32 *= -1.0'f32
        vd3vre32 /= -1.0'f32
        vd3v32 /= -1.0'f32
        md22vre32 /= -1.0'f32
        md22v32 /= -1.0'f32
        check(v3vre32 == v3mare32)
        check(v3v32 == v3ma32)
        check(m22vre32 == m22mare32)
        check(m22v32 == m22ma32)
        check(vd3vre32 == v3mare32)
        check(vd3v32 == v3ma32)
        check(md22vre32 == m22mare32)
        check(md22v32 == m22ma32)

    test "order colMajor rowMajor":
      let
        v22mem = vector(@[1.0, 2.0, 3.0, 4.0]).toComplex
        m22default = v22mem.clone.asMatrix(2, 2) # 13|24
        m22c = v22mem.clone.asMatrix(2, 2, order=colMajor) # 13|24
        m22r = v22mem.clone.asMatrix(2, 2, order=rowMajor) # 12|34

        v6mem23r = vector(@[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]).toComplex
        v6mem32r = vector(@[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).toComplex
        v6mem = vector(@[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).toComplex
        n23c = matrix(@[ # 123456
          @[1.0, 3.0, 5.0],
          @[2.0, 4.0, 6.0]
        ], order=colMajor).toComplex
        n23r = matrix(@[ # 135246
          @[1.0, 3.0, 5.0],
          @[2.0, 4.0, 6.0]
        ], order=rowMajor).toComplex
        n32c = matrix(@[ # 123456
          @[1.0, 4.0],
          @[2.0, 5.0],
          @[3.0, 6.0]
        ], order=colMajor).toComplex
        n32r = matrix(@[ # 142536
          @[1.0, 4.0],
          @[2.0, 5.0],
          @[3.0, 6.0]
        ], order=rowMajor).toComplex

      test "order colMajor rowMajor (float64)":
        check(m22c == m22default)
        check(not (m22c == m22r))
        check(not (m22c =~ m22r))
        check(m22c == m22r.t)
        check(m22c =~ m22r.t)
        check(m22c.sameVector(m22r)) # 1234
        check(m22c.asVector == v22mem) # 1234
        check(m22r.asVector == v22mem) # 1234

        check(n23c.asVector == v6mem) # 123456
        check(n23r.asVector == v6mem23r) # 135246
        check(n32c.asVector == v6mem) # 123456
        check(n32r.asVector == v6mem32r) # 142536

      test "order colMajor rowMajor (float32)":
        let
          v22mem32 = v22mem.to32
          m22default32 = m22default.to32
          m22c32 = m22c.to32
          m22r32 = m22r.to32
          v6mem23r32 = v6mem23r.to32
          v6mem32r32 = v6mem32r.to32
          v6mem32 = v6mem.to32
          n23c32 = n23c.to32
          n23r32 = n23r.to32
          n32c32 = n32c.to32
          n32r32 = n32r.to32

        check(m22c32 == m22default32)
        check(not (m22c32 == m22r32))
        check(not (m22c32 =~ m22r32))
        check(m22c32 == m22r32.t)
        check(m22c32 =~ m22r32.t)
        check(m22c32.sameVector(m22r32)) # 1234
        check(m22c32.asVector == v22mem32) # 1234
        check(m22r32.asVector == v22mem32) # 1234

        check(n23c32.asVector == v6mem32) # 123456
        check(n23r32.asVector == v6mem23r32) # 135246
        check(n32c32.asVector == v6mem32) # 123456
        check(n32r32.asVector == v6mem32r32) # 142536

    test "order and diag of a complex matrix":
      let
        m23c = matrix(@[
          @[1.0, 3.0, 5.0],
          @[2.0, 4.0, 6.0]
        ], order=colMajor).toComplex
        m23r = matrix(@[
          @[1.0, 2.0, 3.0],
          @[4.0, 5.0, 6.0]
        ], order=rowMajor).toComplex

        d22c = vector(@[1.0, 2.0]).diag(order=colMajor).toComplex
        d22r = vector(@[1.0, 2.0]).diag(order=rowMajor).toComplex
        v23cc = vector(@[1.0, 4.0, 3.0, 8.0, 5.0, 12.0]).toComplex
        v23cr = vector(@[1.0, 8.0, 2.0, 10.0, 3.0, 12.0]).toComplex
        v23rc = vector(@[1.0, 4.0, 3.0, 8.0, 5.0, 12.0]).toComplex
        v23rr = vector(@[1.0, 2.0, 3.0, 8.0, 10.0, 12.0]).toComplex

        d33c = vector(@[1.0, 2.0, 3.0]).diag(order=colMajor).toComplex
        d33r = vector(@[1.0, 2.0, 3.0]).diag(order=rowMajor).toComplex
        u23cc = vector(@[1.0, 2.0, 6.0, 8.0, 15.0, 18.0]).toComplex
        u23rc = vector(@[1.0, 4.0, 4.0, 10.0, 9.0, 18.0]).toComplex
        u23cr = vector(@[1.0, 2.0, 6.0, 8.0, 15.0, 18.0]).toComplex
        u23rr = vector(@[1.0, 4.0, 9.0, 4.0, 10.0, 18.0]).toComplex

      test "order and diag of a complex matrix (float64)":
        check(m23c.sameVector(m23r)) # 123456

        var
          ms23c = m23c.clone
          md23c = m23c.clone
          ms23r = m23r.clone
          md23r = m23r.clone
        ms23c *= ci
        md23c /= cmi
        ms23r *= ci
        md23r /= cmi
        check(ms23c.sameVector(md23c))
        check(ms23r.sameVector(md23r))
        check((m23c * ci).sameVector(m23c / cmi))
        check((m23r * ci).sameVector(m23r / cmi))

        check(d22c == d22r)
        check((d22c * m23c).asVector == v23cc) # 1 4 3 8 5 12 colMajor
        check((d22c * m23r).asVector == v23cr) # 1 8 2 10 3 12 colMajor
        check((d22r * m23c).asVector == v23rc) # 1 4 3 8 5 12 colMajor !
        check((d22r * m23r).asVector == v23rr) # 1 2 3 8 10 12 rowMajor
        check((d22c * m23c).sameVector(d22r * m23c)) # 1 4 3 8 5 12
        check(not (d22c * m23r).sameVector(d22r * m23r))

        check(d33c == d33r)
        check((m23c * d33c).asVector == u23cc) # 1 2 6 8 15 18 colMajor
        check((m23r * d33c).asVector == u23rc) # 1 4 4 10 9 18 colMajor !
        check((m23c * d33r).asVector == u23cr) # 1 2 6 8 15 18 colMajor
        check((m23r * d33r).asVector == u23rr) # 1 4 9 4 10 18 rowMajor
        check((m23c * d33c).sameVector(m23c * d33r)) # 1 2 6 8 15 18
        check(not (m23r * d33c).sameVector(m23r * d33r))

      test "order and diag of a complex matrix (float32)":
        let
          m23c32 = m23c.to32
          m23r32 = m23r.to32
          d22c32 = d22c.to32
          d22r32 = d22r.to32
          v23cc32 = v23cc.to32
          v23cr32 = v23cr.to32
          v23rc32 = v23rc.to32
          v23rr32 = v23rr.to32
          d33c32 = d33c.to32
          d33r32 = d33r.to32
          u23cc32 = u23cc.to32
          u23rc32 = u23rc.to32
          u23cr32 = u23cr.to32
          u23rr32 = u23rr.to32

        check(m23c32.sameVector(m23r32)) # 123456

        var
          ms23c32 = m23c32.clone
          md23c32 = m23c32.clone
          ms23r32 = m23r32.clone
          md23r32 = m23r32.clone
        ms23c32 *= ci32
        md23c32 /= cmi32
        ms23r32 *= ci32
        md23r32 /= cmi32
        check(ms23c32.sameVector(md23c32))
        check(ms23r32.sameVector(md23r32))
        check((m23c32 * ci32).sameVector(m23c32 / cmi32))
        check((m23r32 * ci32).sameVector(m23r32 / cmi32))

        check(d22c32 == d22r32)
        check((d22c32 * m23c32).asVector == v23cc32) # 1 4 3 8 5 12 colMajor
        check((d22c32 * m23r32).asVector == v23cr32) # 1 8 2 10 3 12 colMajor
        check((d22r32 * m23c32).asVector == v23rc32) # 1 4 3 8 5 12 colMajor !
        check((d22r32 * m23r32).asVector == v23rr32) # 1 2 3 8 10 12 rowMajor
        check((d22c32 * m23c32).sameVector(d22r32 * m23c32)) # 1 4 3 8 5 12
        check(not (d22c32 * m23r32).sameVector(d22r32 * m23r32))

        check(d33c32 == d33r32)
        check((m23c32 * d33c32).asVector == u23cc32) # 1 2 6 8 15 18 colMajor
        check((m23r32 * d33c32).asVector == u23rc32) # 1 4 4 10 9 18 colMajor !
        check((m23c32 * d33r32).asVector == u23cr32) # 1 2 6 8 15 18 colMajor
        check((m23r32 * d33r32).asVector == u23rr32) # 1 4 9 4 10 18 rowMajor
        check((m23c32 * d33c32).sameVector(m23c32 * d33r32)) # 1 2 6 8 15 18
        check(not (m23r32 * d33c32).sameVector(m23r32 * d33r32))

    test "vdiag and diag of a complex matrix":
      let
        a = matrix(@[
          @[1.0, 4.0],
          @[2.0, 3.0]
        ]).toComplex
        da = vector(@[1.0, 3.0]).toComplex
        b = matrix(@[
          @[0.0, -1.0],
          @[1.0, 0.0]
        ]).toComplex
        db = vector(@[0.0, 0.0]).toComplex
        m = matrix(@[
          @[1.0, 0.0, 1.0],
          @[0.0, 1.0, 2.0]
        ]).toComplex
        dm = vector(@[1.0, 1.0]).toComplex
        n = matrix(@[
          @[1.0, 0.0],
          @[0.0, 1.0],
          @[1.0, 2.0]
        ]).toComplex
        dn = vector(@[1.0, 1.0]).toComplex

      test "vdiag and diag of a complex matrix (float64)":
        check(vdiag(a) =~ da)
        check(vdiag(b) =~ db)
        check(vdiag(m) =~ dm) # TODO: must be assertion error ?
        check(vdiag(n) =~ dn) # TODO: must be assertion error ?
        check(diag(dm) == diag(dn))

      test "vdiag and diag of a complex matrix (float32)":
        check(a.to32.vdiag =~ da.to32)
        check(b.to32.vdiag =~ db.to32)
        check(m.to32.vdiag =~ dm.to32) # TODO: must be assertion error ?
        check(n.to32.vdiag =~ dn.to32) # TODO: must be assertion error ?
        check(dm.to32.diag == dn.to32.diag)

    test "dot scalar and plus minus of a complex vector":
      let
        cri = complex.complex(2.0, -1.0)
        v3cri = vector(@[cri, cri, cri])
        v3re = vector(@[1.0, -2.0, 3.0])
        v3im = vector(@[-1.0, 2.0, -3.0])
        v3 = v3re.toComplex(v3im)
        w3re = vector(@[-1.0, -2.0, -3.0])
        w3im = vector(@[1.0, 2.0, -3.0])
        w3 = w3re.toComplex(w3im)
        vdc3m = complex.complex(6.0, 2.0)
        vdc3mreimz = vdc3m.re.toComplex
        vdu3m = complex.complex(2.0, -6.0)
        vds3mre = vector(@[1.0, -2.0, 3.0])
        vds3mim = vector(@[1.0, -2.0, 3.0])
        vds3m = vds3mre.toComplex(vds3mim)
        vs3mre = vector(@[1.0, -2.0, 3.0])
        vs3mim = vector(@[-3.0, 6.0, -9.0])
        vs3m = vs3mre.toComplex(vs3mim)
        vpw3mre = vector(@[0.0, -4.0, 0.0])
        vpw3mim = vector(@[0.0, 4.0, -6.0])
        vpw3m = vpw3mre.toComplex(vpw3mim)
        vmw3mre = vector(@[2.0, 0.0, 6.0])
        vmw3mim = vector(@[-2.0, 0.0, 0.0])
        vmw3m = vmw3mre.toComplex(vmw3mim)

      test "dot scalar and plus minus of a complex vector (float64)":
        var
          vds3v = v3.clone
          vs3v = v3.clone
          vpw3v = v3.clone
          vmw3v = v3.clone
        vds3v /= cmi
        vs3v *= cri
        vpw3v += w3
        vmw3v -= w3
        check(vds3v == vds3m)
        check(vs3v == vs3m)
        check(vpw3v == vpw3m)
        check(vmw3v == vmw3m)
        check(v3.dotc(v3cri) == vdc3m)
        check(v3.dotc(v3cri) == v3.conjugate.dotu(v3cri))
        check(v3.dotc(v3cri) == v3cri.dotc(v3).conjugate)
        check((v3.dotc(v3cri) + v3cri.dotc(v3)) / 2.0 == vdc3mreimz)
        check(v3.dotu(v3cri) == vdu3m)
        check(v3 * v3cri == vdu3m)
        check(v3 * cri == vs3m)
        check(v3 / cmi == vds3m)
        check(v3 + w3 == vpw3m)
        check(v3 - w3 == vmw3m)

      test "dot scalar and plus minus of a complex vector (float32)":
        let
          cri32 = cri.to32
          v3cri32 = v3cri.to32
          v332 = v3.to32
          w332 = w3.to32
          vdc3m32 = vdc3m.to32
          vdc3mreimz32 = vdc3mreimz.to32
          vdu3m32 = vdu3m.to32
          vds3m32 = vds3m.to32
          vs3m32 = vs3m.to32
          vpw3m32 = vpw3m.to32
          vmw3m32 = vmw3m.to32
        var
          vds3v32 = v332.clone
          vs3v32 = v332.clone
          vpw3v32 = v332.clone
          vmw3v32 = v332.clone
        vds3v32 /= cmi32
        vs3v32 *= cri32
        vpw3v32 += w332
        vmw3v32 -= w332
        check(vds3v32 == vds3m32)
        check(vs3v32 == vs3m32)
        check(vpw3v32 == vpw3m32)
        check(vmw3v32 == vmw3m32)
        check(v332.dotc(v3cri32) == vdc3m32)
        check(v332.dotc(v3cri32) == v332.conjugate.dotu(v3cri32))
        check(v332.dotc(v3cri32) == v3cri32.dotc(v332).conjugate)
        check((v332.dotc(v3cri32) + v3cri32.dotc(v332)) / 2.0 == vdc3mreimz32)
        check(v332.dotu(v3cri32) == vdu3m32)
        check(v332 * v3cri32 == vdu3m32)
        check(v332 * cri32 == vs3m32)
        check(v332 / cmi32 == vds3m32)
        check(v332 + w332 == vpw3m32)
        check(v332 - w332 == vmw3m32)

    test "order and plus minus of a complex matrix":
      let
        p23cim = matrix(@[
          @[-1.0, -3.0, -5.0],
          @[-2.0, -4.0, -6.0]
        ], order=colMajor)
        p23c = matrix(@[
          @[1.0, 3.0, 5.0],
          @[2.0, 4.0, 6.0]
        ], order=colMajor).toComplex(p23cim)
        p23rim = matrix(@[
          @[-1.0, -2.0, -3.0],
          @[-4.0, -5.0, -6.0]
        ], order=rowMajor)
        p23r = matrix(@[
          @[1.0, 2.0, 3.0],
          @[4.0, 5.0, 6.0]
        ], order=rowMajor).toComplex(p23rim)
        v23mre = vector(@[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v23mim = vector(@[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
        v23m = v23mre.toComplex(v23mim) # as memory
        v23cre = vector(@[2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        v23cim = vector(@[-2.0, -4.0, -6.0, -8.0, -10.0, -12.0])
        v23c = v23cre.toComplex(v23cim) # p23r as colMajor
        v23rre = vector(@[2.0, 6.0, 5.0, 9.0, 8.0, 12.0])
        v23rim = vector(@[-2.0, -6.0, -5.0, -9.0, -8.0, -12.0])
        v23r = v23rre.toComplex(v23rim) # p23r as rowMajor
        v23vre = vector(@[2.0, 5.0, 8.0, 6.0, 9.0, 12.0])
        v23vim = vector(@[-2.0, -5.0, -8.0, -6.0, -9.0, -12.0])
        v23v = v23vre.toComplex(v23vim) # p23c as rowMajor

        p23zim = matrix(@[
          @[-1.0, 2.0, -3.0],
          @[4.0, -5.0, -6.0]
        ], order=rowMajor)
        p23z = matrix(@[
          @[1.0, -2.0, 3.0],
          @[-4.0, 5.0, -6.0]
        ], order=colMajor).toComplex(p23zim)
        v23cpzre = vector(@[2.0, -2.0, 1.0, 9.0, 8.0, 0.0])
        v23cpzim = vector(@[-2.0, 2.0, -1.0, -9.0, -8.0, -12.0])
        v23cpz = v23cpzre.toComplex(v23cpzim)
        v23rpzre = vector(@[2.0, 0.0, 6.0, 0.0, 10.0, 0.0])
        v23rpzim = vector(@[-2.0, 0.0, -6.0, 0.0, -10.0, -12.0])
        v23rpz = v23rpzre.toComplex(v23rpzim)
        v23cmzre = vector(@[0.0, 6.0, 5.0, -1.0, 2.0, 12.0])
        v23cmzim = vector(@[0.0, -6.0, -5.0, 1.0, -2.0, 0.0])
        v23cmz = v23cmzre.toComplex(v23cmzim)
        v23rmzre = vector(@[0.0, 4.0, 0.0, 8.0, 0.0, 12.0])
        v23rmzim = vector(@[0.0, -4.0, 0.0, -8.0, 0.0, 0.0])
        v23rmz = v23rmzre.toComplex(v23rmzim)

      test "order and plus minus of a complex matrix (float64)":
        check(p23c.asVector == v23m)
        check(p23r.asVector == v23m)
        check(not ((p23c + p23r).asVector == v23c))
        check((p23c + p23r).asVector == v23r)
        check((p23r + p23c).asVector == v23v)
        check((p23r + p23r).asVector == v23c)
        check((p23c + p23c).asVector == v23c)

        var
          o23cr = p23c.clone
          o23rc = p23r.clone
          o23rr = p23r.clone
          o23cc = p23c.clone
        o23cr += p23r
        o23rc += p23c
        o23rr += p23r
        o23cc += p23c
        check(o23cr.asVector == v23r)
        check(o23rc.asVector == v23v)
        check(o23rr.asVector == v23c)
        check(o23cc.asVector == v23c)

        check((p23c + p23z).asVector == v23cpz)
        check((p23r + p23z).asVector == v23rpz)
        check((p23c - p23z).asVector == v23cmz)
        check((p23r - p23z).asVector == v23rmz)

        var
          o23cpz = p23c.clone
          o23rpz = p23r.clone
          o23cmz = p23c.clone
          o23rmz = p23r.clone
        o23cpz += p23z
        o23rpz += p23z
        o23cmz -= p23z
        o23rmz -= p23z
        check(o23cpz.asVector == v23cpz)
        check(o23rpz.asVector == v23rpz)
        check(o23cmz.asVector == v23cmz)
        check(o23rmz.asVector == v23rmz)

      test "order and plus minus of a complex matrix (float32)":
        let
          p23c32 = p23c.to32
          p23r32 = p23r.to32
          v23m32 = v23m.to32
          v23c32 = v23c.to32
          v23r32 = v23r.to32
          v23v32 = v23v.to32

          p23z32 = p23z.to32
          v23cpz32 = v23cpz.to32
          v23rpz32 = v23rpz.to32
          v23cmz32 = v23cmz.to32
          v23rmz32 = v23rmz.to32

        check(p23c32.asVector == v23m32)
        check(p23r32.asVector == v23m32)
        check(not ((p23c32 + p23r32).asVector == v23c32))
        check((p23c32 + p23r32).asVector == v23r32)
        check((p23r32 + p23c32).asVector == v23v32)
        check((p23r32 + p23r32).asVector == v23c32)
        check((p23c32 + p23c32).asVector == v23c32)

        var
          o23cr32 = p23c32.clone
          o23rc32 = p23r32.clone
          o23rr32 = p23r32.clone
          o23cc32 = p23c32.clone
        o23cr32 += p23r32
        o23rc32 += p23c32
        o23rr32 += p23r32
        o23cc32 += p23c32
        check(o23cr32.asVector == v23r32)
        check(o23rc32.asVector == v23v32)
        check(o23rr32.asVector == v23c32)
        check(o23cc32.asVector == v23c32)

        check((p23c32 + p23z32).asVector == v23cpz32)
        check((p23r32 + p23z32).asVector == v23rpz32)
        check((p23c32 - p23z32).asVector == v23cmz32)
        check((p23r32 - p23z32).asVector == v23rmz32)

        var
          o23cpz32 = p23c32.clone
          o23rpz32 = p23r32.clone
          o23cmz32 = p23c32.clone
          o23rmz32 = p23r32.clone
        o23cpz32 += p23z32
        o23rpz32 += p23z32
        o23cmz32 -= p23z32
        o23rmz32 -= p23z32
        check(o23cpz32.asVector == v23cpz32)
        check(o23rpz32.asVector == v23rpz32)
        check(o23cmz32.asVector == v23cmz32)
        check(o23rmz32.asVector == v23rmz32)

run()
