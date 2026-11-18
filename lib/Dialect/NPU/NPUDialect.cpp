#include "NPU/NPUDialect.h"
#include "NPU/NPUOps.h"

#include "NPU/NPUOpsDialect.cpp.inc"

void mlir::npu::NPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NPU/NPUOps.cpp.inc"
      >();
}
