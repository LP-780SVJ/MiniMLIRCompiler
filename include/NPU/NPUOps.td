#ifndef NPU_OPS_TD
#define NPU_OPS_TD

include "NPU/NPUDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MatMulOp : Op<NPU_Dialect, "matmul", [Pure]> {
  let summary = "Virtual NPU FP32 matrix multiplication";
  let description = [{
    Computes the complete tensor result [M,N] from [M,K] and [K,N].
    Dimensions are positive. Tiling, accumulator initialization, SRAM
    placement and DMA are handled by later lowering stages.
  }];
  let arguments = (ins
    StaticShapeTensorOf<[F32]>:$lhs,
    StaticShapeTensorOf<[F32]>:$rhs
  );
  let results = (outs StaticShapeTensorOf<[F32]>:$result);
  let hasVerifier = 1;
}

def TiledMatMulOp : Op<NPU_Dialect, "tiled_matmul", [Pure]> {
  let summary = "Full FP32 MatMul with a static Virtual NPU tile plan";
  let arguments = (ins
    StaticShapeTensorOf<[F32]>:$lhs,
    StaticShapeTensorOf<[F32]>:$rhs,
    StrAttr:$target,
    I64Attr:$tile_m,
    I64Attr:$tile_n,
    I64Attr:$tile_k,
    ArrayAttr:$tiles
  );
  let results = (outs StaticShapeTensorOf<[F32]>:$result);
  let hasVerifier = 1;
}

def SRAMPlannedMatMulOp : Op<NPU_Dialect, "sram_planned_matmul", [Pure]> {
  let summary = "Full FP32 MatMul with verified Tile and SRAM plans";
  let arguments = (ins
    StaticShapeTensorOf<[F32]>:$lhs,
    StaticShapeTensorOf<[F32]>:$rhs,
    StrAttr:$target,
    I64Attr:$tile_m,
    I64Attr:$tile_n,
    I64Attr:$tile_k,
    ArrayAttr:$tiles,
    DictionaryAttr:$memory_plan
  );
  let results = (outs StaticShapeTensorOf<[F32]>:$result);
  let hasVerifier = 1;
}

#endif
