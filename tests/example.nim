import nimat

when isMainModule:
  let a = eye(3)
  echo a

  var
    rows = @[0'i32, 3, 4, 7, 9]
    cols = @[0'i32, 2, 3, 1, 0, 2, 3, 1, 3]
    vals = @[1'f32, 2, 3, 4, 5, 6, 7, 8, 9]
  let x = csr(rows, cols, vals, numCols = 4)
  let y = x.gpu()
  let z = y.toCoo()
  let w = z.toCsr().cpu()
  echo x[]
  echo w[]