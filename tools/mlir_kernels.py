"""Emit packed-W4 numerical loops in standard MLIR dialects.

These are restricted operator templates, not a general torch graph compiler.
The input contract is validated by export_linear.py before this emitter is used.
"""

UNPACK = """
          %kw = arith.divui %k, %c2 : index
          %wi = arith.addi %row_w, %kw : index
          %packed = memref.load %w[%wi] : memref<?xi8>
          %half = arith.remui %k, %c2 : index
          %shift_index = arith.muli %half, %c4 : index
          %shift = arith.index_cast %shift_index : index to i8
          %shifted = arith.shrui %packed, %shift : i8
          %nibble = arith.andi %shifted, %mask : i8
          %signed_high = arith.shli %nibble, %four : i8
          %signed_q = arith.shrsi %signed_high, %four : i8
          %q = arith.extsi %signed_q : i8 to i32
"""


def emit_kernels():
    common = """
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %four = arith.constant 4 : i8
    %mask = arith.constant 15 : i8
    %zero = arith.constant 0.0 : f32
    %kp1 = arith.addi %K, %c1 : index
    %stride_w = arith.divui %kp1, %c2 : index
    %gm1 = arith.subi %G, %c1 : index
    %kg = arith.addi %K, %gm1 : index
    %groups = arith.divui %kg, %G : index
"""
    bf16 = """
  func.func @nm_bf16(%x: memref<?xi16>, %w: memref<?xi8>, %scales: memref<?xf32>,
                     %y: memref<?xf32>, %M: index, %N: index, %K: index, %G: index)
      attributes {llvm.emit_c_interface, nm.variant_id = 1 : i32, nm.weight_format = "packed_w4"} {
""" + common + """
    %sixteen = arith.constant 16 : i32
    scf.for %m = %c0 to %M step %c1 {
      %row_x = arith.muli %m, %K : index
      %row_y = arith.muli %m, %N : index
      scf.for %n = %c0 to %N step %c1 {
        %row_w = arith.muli %n, %stride_w : index
        %row_s = arith.muli %n, %groups : index
        %sum = scf.for %k = %c0 to %K step %c1 iter_args(%acc = %zero) -> (f32) {
""" + UNPACK + """
          %group = arith.divui %k, %G : index
          %si = arith.addi %row_s, %group : index
          %scale = memref.load %scales[%si] : memref<?xf32>
          %qf = arith.sitofp %q : i32 to f32
          %wf = arith.mulf %qf, %scale : f32
          %xi = arith.addi %row_x, %k : index
          %bf = memref.load %x[%xi] : memref<?xi16>
          %bits16 = arith.extui %bf : i16 to i32
          %bits32 = arith.shli %bits16, %sixteen : i32
          %xf = arith.bitcast %bits32 : i32 to f32
          %product = arith.mulf %xf, %wf : f32
          %next = arith.addf %acc, %product : f32
          scf.yield %next : f32
        }
        %yi = arith.addi %row_y, %n : index
        memref.store %sum, %y[%yi] : memref<?xf32>
      }
    }
    return
  }
"""
    a8 = """
  func.func @nm_a8(%x: memref<?xi8>, %w: memref<?xi8>, %scales: memref<?xf32>,
                   %y: memref<?xf32>, %M: index, %N: index, %K: index, %G: index, %activation_scale: f32)
      attributes {llvm.emit_c_interface, nm.variant_id = 2 : i32, nm.weight_format = "packed_w4"} {
""" + common + """
    %zero_i = arith.constant 0 : i32
    scf.for %m = %c0 to %M step %c1 {
      %row_x = arith.muli %m, %K : index
      %row_y = arith.muli %m, %N : index
      scf.for %n = %c0 to %N step %c1 {
        %row_w = arith.muli %n, %stride_w : index
        %row_s = arith.muli %n, %groups : index
        %sum = scf.for %g = %c0 to %groups step %c1 iter_args(%acc = %zero) -> (f32) {
          %begin = arith.muli %g, %G : index
          %limit = arith.addi %begin, %G : index
          %end = arith.minui %limit, %K : index
          %dot = scf.for %k = %begin to %end step %c1 iter_args(%integer_acc = %zero_i) -> (i32) {
""" + UNPACK + """
            %xi = arith.addi %row_x, %k : index
            %aq = memref.load %x[%xi] : memref<?xi8>
            %a = arith.extsi %aq : i8 to i32
            %product = arith.muli %a, %q : i32
            %next = arith.addi %integer_acc, %product : i32
            scf.yield %next : i32
          }
          %si = arith.addi %row_s, %g : index
          %scale = memref.load %scales[%si] : memref<?xf32>
          %dot_f = arith.sitofp %dot : i32 to f32
          %activation_scaled = arith.mulf %dot_f, %activation_scale : f32
          %scaled = arith.mulf %activation_scaled, %scale : f32
          %next_f = arith.addf %acc, %scaled : f32
          scf.yield %next_f : f32
        }
        %yi = arith.addi %row_y, %n : index
        memref.store %sum, %y[%yi] : memref<?xf32>
      }
    }
    return
  }
"""
    return "// Generated packed W4 executors; arithmetic/storage semantics are explicit.\nmodule {\n" + bf16 + a8 + "}\n"
