# complex linear algebra solvers

import macros
import complex
include nimlapack # (not use import) private type lapack_complex_[float|double]

## cast for [c|z]geev [c|z]gesv in nimlapack.nim
## to fix .fp ambiguous between ptr lapack_complex_[float|double]
##  Vector or Matrix[Complex[float32 | float64]]

template force_cast_ptr*[P: complex.Complex[float32]](p: ptr P):
  ptr lapack_complex_float=
  cast[ptr lapack_complex_float](p)

template force_cast_ptr*[P: complex.Complex[float64]](p: ptr P):
  ptr lapack_complex_double=
  cast[ptr lapack_complex_double](p)

template force_cast_ptr*[P: float32](p: ptr P):
  ptr cfloat=
  cast[ptr cfloat](p)

template force_cast_ptr*[P: float64](p: ptr P):
  ptr cdouble=
  cast[ptr cdouble](p)

template force_cast_ptr*[T, P](p: ptr P): ptr T=
  cast[ptr T](p)

proc czgetAddress(n: NimNode): NimNode =
  let t = n.getTypeImpl
  if t.kind == nnkBracketExpr and $(t[0]) == "seq":
    result = quote do:
      addr `n`[0]
  elif t.kind == nnkRefTy and $(t[0]) in ["Vector", "Vector:ObjectType"]:
    result = quote do:
      force_cast_ptr(`n`.fp)
  elif t.kind == nnkRefTy and $(t[0]) in ["Matrix", "Matrix:ObjectType"]:
    result = quote do:
      force_cast_ptr(`n`.fp)
  elif $t == "cstring":
    result = quote do:
      `n`
  else:
    result = quote do:
      addr `n`

macro czfortran*(f: untyped, callArgs: varargs[typed]): auto =
  var transformedCallArgs = newSeqOfCap[NimNode](callArgs.len)
  for x in callArgs:
    transformedCallArgs.add(czgetAddress(x))
  result = newCall(f, transformedCallArgs)
