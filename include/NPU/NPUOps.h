#ifndef NPU_NPUOPS_H
#define NPU_NPUOPS_H

#include "NPU/NPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "NPU/NPUOps.h.inc"

#endif
