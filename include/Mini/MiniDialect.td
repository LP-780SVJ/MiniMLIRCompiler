#ifndef MINI_DIALECT_TD
#define MINI_DIALECT_TD

include "mlir/IR/OpBase.td"

def Mini_Dialect : Dialect {
  let name = "mini";
  let cppNamespace = "::mlir::mini";
  let summary = "Static FP32 tensor operations for MiniMLIRCompiler";
  let description = [{
    Mini is the high-level tensor dialect of MiniMLIRCompiler. Operations
    are introduced in Day 9; this file establishes its identity and C++ API.
  }];
}

#endif // MINI_DIALECT_TD
