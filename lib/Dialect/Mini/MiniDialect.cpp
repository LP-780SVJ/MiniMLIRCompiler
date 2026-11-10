
#include "Mini/MiniDialect.h"
#include "Mini/MiniOps.h"

using namespace mlir;
using namespace mlir::mini;

// TableGen 生成的 Dialect 类实现
#include "Mini/MiniOpsDialect.cpp.inc"

void MiniDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Mini/MiniOps.cpp.inc"
  >();
}
