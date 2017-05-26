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

########################### GESV #################################
#n is number of rows of A and B that are being used
#nrhs is number of columns of matrix B
#a is the matrix to be solved in data vector form (should be column-major)
#lda is number of rows of A
#ipvt is an integer array of size n
#b is the matrix B which gets filled with the X values
#ldb is the number of rows of B
#info is an integer with return OK or not
when defined(windows):
  const lapackSuffix = ".dll"
elif defined(macosx):
  const lapackSuffix = ".dylib"
else:
  const lapackSuffix = ".so.(|3|2|1|0)"

const lapackPrefix = "liblapack"
const lapackName = lapackPrefix & lapackSuffix

proc gesv*(
  n: ptr cint,
  nrhs: ptr cint,
  a: ptr cfloat,
  lda: ptr cint,
  ipvt: ptr cint,
  b: ptr cfloat,
  ldb: ptr cint,
  info: ptr cint
  ) {.cdecl, importc: "sgesv_", dynlib: lapackName.}

proc gesv*(
  n: ptr cint,
  nrhs: ptr cint,
  a: ptr cdouble,
  lda: ptr cint,
  ipvt: ptr cint,
  b: ptr cdouble,
  ldb: ptr cint,
  info: ptr cint
  ) {.cdecl, importc: "dgesv_", dynlib: lapackName.}
