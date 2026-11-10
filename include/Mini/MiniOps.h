
#ifndef MINI_MINIOPS_H
#define MINI_MINIOPS_H

#include "mlir/IR/OpDefinition.h"
#include "Mini/MiniDialect.h"

// TableGen 生成的 Operation 类声明
#define GET_OP_CLASSES
#include "Mini/MiniOps.h.inc"

#endif // MINI_MINIOPS_H
