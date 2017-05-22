import sequtils
import nimcuda/[cuda_runtime_api, driver_types, cusparse, nimcuda]

type
  CArray{.unchecked.}[T] = array[1, T]
  CPointer[T] = ptr CArray[T]

proc allocCPointer[T](n: Natural): CPointer[T] {.inline.} =
  cast[CPointer[T]](alloc(n * sizeof(T)))

proc first[T](p: CPointer[T]): ptr T {.inline.} = addr(p[0])

proc first[T](a: var seq[T]): ptr T {.inline.} = addr(a[0])

template pointerTo(x: untyped) = cast[ptr pointer](addr x)

template size[A](xs: seq[A]): int = xs.len * sizeof(A)

type
  Complex*[A] = tuple[re, im: A]
  Number* = float32 or float64 or Complex[float32] or Complex[float64]
  SparseMatrixKind* = enum
    CSR, CSC, COO
  SparseMatrixObj*[A] = object
    kind*: SparseMatrixKind
    M*, N*, nnz*: int32
    rows*, cols*: ptr int32
    vals*: ptr A
  SparseMatrix*[A] = ref SparseMatrixObj[A]
  SparseMatrixGpuObj*[A] = object
    kind*: SparseMatrixKind
    M*, N*, nnz*: int32
    rows*, cols*: ptr int32
    vals*: ptr A
  SparseMatrixGpu*[A] = ref SparseMatrixGpuObj[A]

proc dealloc[A](m: SparseMatrix[A]) =
  if m.rows != nil: dealloc(m.rows)
  if m.cols != nil: dealloc(m.cols)
  if m.vals != nil: dealloc(m.vals)

proc dealloc[A](m: SparseMatrixGpu[A]) =
  if m.rows != nil: check cudaFree(m.rows)
  if m.cols != nil: check cudaFree(m.cols)
  if m.vals != nil: check cudaFree(m.vals)

proc rowLen*(m: SparseMatrix or SparseMatrixGpu): int32 =
  case m.kind
  of CSR: m.M + 1
  of CSC, COO: m.nnz

proc colLen*(m: SparseMatrix or SparseMatrixGpu): int32 =
  case m.kind
  of CSR, COO: m.nnz
  of CSC: m.N + 1

proc sizes[A](m: SparseMatrix[A] or SparseMatrixGpu[A]): tuple[r, c, v: int32] =
  (
    (m.rowLen * sizeof(int32)).int32,
    (m.colLen * sizeof(int32)).int32,
    (m.nnz * sizeof(A)).int32
  )

proc sparse*[A: Number](kind: SparseMatrixKind, M, N, nnz: int32, rows, cols: var seq[int32], vals: var seq[A]): SparseMatrix[A] =
  new result, dealloc
  result.kind = kind
  result.M = N
  result.N = N
  result.nnz = nnz
  let (r, c, v) = result.sizes
  result.rows = cast[ptr int32](alloc(r))
  result.cols = cast[ptr int32](alloc(c))
  result.vals = cast[ptr A](alloc(v))
  copyMem(result.rows, rows.first, r)
  copyMem(result.cols, cols.first, c)
  copyMem(result.vals, vals.first, v)

proc csr*[A: Number](rows, cols: var seq[int32], vals: var seq[A], numCols: int32): SparseMatrix[A] =
  sparse(CSR, rows.len.int32, numCols, vals.len.int32, rows, cols, vals)

proc csc*[A: Number](rows, cols: var seq[int32], vals: var seq[A], numRows: int32): SparseMatrix[A] =
  sparse(CSC, numRows, cols.len.int32, vals.len.int32, rows, cols, vals)

proc coo*[A: Number](rows, cols: var seq[int32], vals: var seq[A], numRows, numCols: int32): SparseMatrix[A] =
  sparse(COO, numRows, numCols, vals.len.int32, rows, cols, vals)

var defaultHandle: cusparseHandle_t
check cusparseCreate(addr defaultHandle)

const idx = CUSPARSE_INDEX_BASE_ZERO

template allocateAll(x, r, c, v: untyped) =
  check cudaMalloc(pointerTo x.rows, r)
  check cudaMalloc(pointerTo x.cols, c)
  check cudaMalloc(pointerTo x.vals, v)

proc gpu*[A: Number](m: SparseMatrix[A]): SparseMatrixGpu[A] =
  new result, dealloc
  result.kind = m.kind
  result.M = m.M
  result.N = m.N
  result.nnz = m.nnz
  let (r, c, v) = result.sizes
  check cudaMalloc(pointerTo result.rows, r)
  check cudaMalloc(pointerTo result.cols, c)
  check cudaMalloc(pointerTo result.vals, v)
  check cudaMemcpy(result.rows, m.rows, r, cudaMemcpyHostToDevice)
  check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyHostToDevice)
  check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyHostToDevice)

proc cpu*[A: Number](m: SparseMatrixGpu[A]): SparseMatrix[A] =
  new result, dealloc
  result.kind = m.kind
  result.M = m.M
  result.N = m.N
  result.nnz = m.nnz
  let (r, c, v) = result.sizes
  result.rows = cast[ptr int32](alloc(r))
  result.cols = cast[ptr int32](alloc(c))
  result.vals = cast[ptr A](alloc(v))
  check cudaMemcpy(result.rows, m.rows, r, cudaMemcpyDeviceToHost)
  check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToHost)
  check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToHost)

proc toCsr*[A: Number](m: SparseMatrixGpu[A], handle = defaultHandle): SparseMatrixGpu[A] =
  new result, dealloc
  result.kind = CSR
  result.M = m.M
  result.N = m.N
  result.nnz = m.nnz
  let (r, c, v) = result.sizes
  allocateAll(result, r, c, v)
  case m.kind
  of CSR:
    check cudaMemcpy(result.rows, m.rows, r, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)
  of CSC:
    when A is float32:
      check cusparseScsr2csc(handle, m.N, m.M, m.nnz, result.vals, result.cols, result.rows, m.vals, m.cols, m.rows, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is float64:
      check cusparseDcsr2csc(handle, m.N, m.M, m.nnz, result.vals, result.cols, result.rows, m.vals, m.cols, m.rows, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is Complex[float32]:
      check cusparseCcsr2csc(handle, m.N, m.M, m.nnz, result.vals, result.cols, result.rows, m.vals, m.cols, m.rows, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is Complex[float64]:
      check cusparseZcsr2csc(handle, m.N, m.M, m.nnz, result.vals, result.cols, result.rows, m.vals, m.cols, m.rows, CUSPARSE_ACTION_NUMERIC, idx)
  of COO:
    check cusparseXcoo2csr(handle, m.rows, m.nnz, m.N, result.rows, idx)
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)

proc toCsc*[A: Number](m: SparseMatrixGpu[A], handle = defaultHandle): SparseMatrixGpu[A] =
  case m.kind
  of CSR:
    new result, dealloc
    result.kind = CSC
    result.M = m.M
    result.N = m.N
    result.nnz = m.nnz
    let (r, c, v) = result.sizes
    allocateAll(result, r, c, v)
    when A is float32:
      check cusparseScsr2csc(handle, m.M, m.N, m.nnz, result.vals, result.rows, result.cols, m.vals, m.rows, m.cols, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is float64:
      check cusparseDcsr2csc(handle, m.M, m.N, m.nnz, result.vals, result.rows, result.cols, m.vals, m.rows, m.cols, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is Complex[float32]:
      check cusparseCcsr2csc(handle, m.M, m.N, m.nnz, result.vals, result.rows, result.cols, m.vals, m.rows, m.cols, CUSPARSE_ACTION_NUMERIC, idx)
    elif A is Complex[float64]:
      check cusparseZcsr2csc(handle, m.M, m.N, m.nnz, result.vals, result.rows, result.cols, m.vals, m.rows, m.cols, CUSPARSE_ACTION_NUMERIC, idx)
  of CSC:
    new result, dealloc
    result.kind = CSC
    result.M = m.M
    result.N = m.N
    result.nnz = m.nnz
    let (r, c, v) = result.sizes
    allocateAll(result, r, c, v)
    check cudaMemcpy(result.rows, m.rows, r, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)
  of COO:
    result = m.toCsr().toCsc()

proc toCoo*[A: Number](m: SparseMatrixGpu[A], handle = defaultHandle): SparseMatrixGpu[A] =
  new result, dealloc
  result.kind = COO
  result.M = m.M
  result.N = m.N
  result.nnz = m.nnz
  let (r, c, v) = result.sizes
  allocateAll(result, r, c, v)
  case m.kind
  of CSR:
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)
    check cusparseXcsr2coo(handle, m.rows, m.nnz, m.M, result.rows, idx)
  of CSC:
    result = m.toCsr().toCoo()
  of COO:
    check cudaMemcpy(result.rows, m.rows, r, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.cols, m.cols, c, cudaMemcpyDeviceToDevice)
    check cudaMemcpy(result.vals, m.vals, v, cudaMemcpyDeviceToDevice)