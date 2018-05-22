# Copyright 2017 UniCredit S.p.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import macros, sequtils, strutils

template pointerTo*(x: untyped) = cast[ptr pointer](addr x)

proc first*[T](a: var seq[T]): ptr T {.inline.} = addr(a[0])

macro overload*(s: untyped, p: typed): auto =
  let args = p.getTypeImpl[0]
  var j = 0
  for c in args.children:
    if j > 0:
      if $(c[0]) == "result":
        c[0] = genSym(nskParam, "res")
    inc j
  var
    params = toSeq(args.children)
    callArgs = newSeq[NimNode]()
    i = 0
  for c in args.children:
    if i > 0:
      callArgs.add(c[0])
    inc i
  let
    call = newCall(p, callArgs)
    overloadedProc = newProc(
      name = s,
      params = params,
      body = newStmtList(call)
    )
  result = newStmtList(overloadedProc)

template overload*(s: untyped, p, q: typed) =
  overload(s, p)
  overload(s, q)

proc getAddress(n: NimNode): NimNode =
  let t = n.getTypeImpl
  if t.kind == nnkBracketExpr and $(t[0]) == "seq":
    result = quote do:
      addr `n`[0]
  elif t.kind == nnkRefTy and $(t[0]) == "Vector:ObjectType":
    result = quote do:
      `n`.fp
  elif t.kind == nnkRefTy and $(t[0]) == "Matrix:ObjectType":
    result = quote do:
      `n`.fp
  else:
    result = quote do:
      addr `n`

macro fortran*(f: untyped, callArgs: varargs[typed]): auto =
  var transformedCallArgs = newSeqOfCap[NimNode](callArgs.len)
  for x in callArgs:
    transformedCallArgs.add(getAddress(x))
  result = newCall(f, transformedCallArgs)