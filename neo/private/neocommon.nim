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
import macros, sequtils

template pointerTo*(x: untyped) = cast[ptr pointer](addr x)

proc first*[T](a: var seq[T]): ptr T {.inline.} = addr(a[0])

macro overload*(s: untyped, p: typed): auto =
  let args = p.getTypeImpl[0]
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