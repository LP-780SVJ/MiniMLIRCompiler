#ifndef NPU_DIALECT_TD
#define NPU_DIALECT_TD

include "mlir/IR/OpBase.td"

def NPU_Dialect : Dialect {
  let name = "npu";
  let cppNamespace = "::mlir::npu";
  let summary = "Compute operations targeting Virtual NPU v1";
}

#endif
