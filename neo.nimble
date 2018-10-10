mode = ScriptMode.Verbose

packageName   = "neo"
version       = "0.2.3"
author        = "Andrea Ferretti"
description   = "Linear Algebra for Nim"
license       = "Apache2"
skipDirs      = @["tests", "benchmarks", "htmldocs"]
skipFiles     = @["_config.yml"]

requires "nim >= 0.18.0", "nimblas >= 0.2.0", "nimcuda >= 0.1.4",
  "nimlapack >= 0.1.1"

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

task teststatic, "run CPU static tests":
  configForTests()
  setCommand "c", "tests/tstatics.nim"

task testshared, "run CPU shared heap tests":
  configForTests()
  --threads:on
  setCommand "c", "tests/tshared.nim"

task testopenblas, "run CPU tests on openblas":
  configForTests()
  --define:"blas=openblas"
  --define:"lapack=openblas"
  setCommand "c", "tests/all.nim"

task testmkl, "run CPU tests on mkl":
  configForTests()
  --define:"blas=mkl_intel_lp64"
  --clibdir: "/opt/intel/mkl/lib/intel64"
  --passl: "/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a"
  --passl: "-lmkl_core"
  --passl: "-lmkl_sequential"
  --passl: "-lpthread"
  --passl: "-lm"
  --dynlibOverride:mkl_intel_lp64
  setCommand "c", "tests/all.nim"

task compilecuda, "only compile GPU tests (when not having a GPU)":
  --hints: off
  --linedir: on
  --stacktrace: on
  --linetrace: on
  --debuginfo
  --path: "."
  --compileOnly
  setCommand "c", "tests/allcuda.nim"

task testcuda, "run GPU tests":
  configForTests()
  --gc:markAndSweep # TODO: remove temporary workaround
  setCommand "c", "tests/allcuda.nim"

task testcudadense, "run GPU dense tests":
  configForTests()
  setCommand "c", "tests/tcudadense.nim"

task testcudasparse, "run GPU sparse tests":
  configForTests()
  setCommand "c", "tests/tcudasparse.nim"

task testrw, "run tests for rewrite macros":
  configForTests()
  --define:neoCountRewrites
  setCommand "c", "tests/rewrites.nim"

task benchmark, "run CPU benchmarks":
  configForBenchmarks()
  setCommand "c", "benchmarks/bench_cpu.nim"

task benchmarkcuda, "run GPU benchmarks":
  configForBenchmarks()
  setCommand "c", "benchmarks/bench_gpu.nim"

task docs, "generate documentation":
  exec("mkdir -p htmldocs/neo")
  --project
  --docSeeSrcUrl: "https://github.com/unicredit/neo/blob/master"
  setCommand "doc2", "neo.nim"