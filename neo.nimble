mode = ScriptMode.Verbose

packageName   = "neo"
version       = "0.1.0"
author        = "Andrea Ferretti"
description   = "Linear Algebra for Nim"
license       = "Apache2"
skipDirs      = @["tests", "bench", "htmldocs"]
skipFiles     = @["_config.yml"]

requires "nim >= 0.17.0", "nimblas >= 0.1.3", "nimcuda >= 0.1.2",
  "nimlapack >= 0.1.0"

--forceBuild

when defined(nimdistros):
  import distros
  if detectOs(Ubuntu) or detectOs(Debian):
    foreignDep "libblas-dev"
    foreignDep "libopenblas-dev"
    foreignDep "liblapack-dev"
  else:
    foreignDep "libblas"
    foreignDep "liblapack"

proc configForTests() =
  --hints: off
  --linedir: on
  --stacktrace: on
  --linetrace: on
  --debuginfo
  --path: "."
  --run

proc configForBenchmarks() =
  --define: release
  --path: "."
  --run

task test, "run CPU tests":
  configForTests()
  setCommand "c", "tests/all.nim"

task testdense, "run CPU dense tests":
  configForTests()
  setCommand "c", "tests/tdense.nim"

task testsparse, "run CPU sparse tests":
  configForTests()
  setCommand "c", "tests/tsparse.nim"

task testopenblas, "run CPU tests on openblas":
  configForTests()
  --define: openblas
  setCommand "c", "tests/all.nim"

task testmkl, "run CPU tests on mkl":
  configForTests()
  --dynlibOverride:mkl_intel_lp64
  --passL:"/home/papillon/.intel/mkl/lib/intel64/libmkl_intel_lp64.a"
  --define: mkl
  setCommand "c", "tests/all.nim"

task testcuda, "run GPU tests":
  configForTests()
  setCommand "c", "tests/allcuda.nim"

task testcudadense, "run GPU dense tests":
  configForTests()
  setCommand "c", "tests/tcudadense.nim"

task testcudasparse, "run GPU sparse tests":
  configForTests()
  setCommand "c", "tests/tcudasparse.nim"

task docs, "generate documentation":
  exec("mkdir -p htmldocs/neo")
  --project
  --docSeeSrcUrl: "https://github.com/unicredit/neo/blob/master"
  setCommand "doc2", "neo.nim"