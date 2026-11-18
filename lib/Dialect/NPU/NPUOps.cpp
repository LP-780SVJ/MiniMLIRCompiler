#include "NPU/NPUOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"

mlir::LogicalResult mlir::npu::MatMulOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto result = getResult().getType();

  if (lhs.getRank() != 2 || rhs.getRank() != 2 || result.getRank() != 2)
    return emitOpError() << "requires rank-2 tensors";

  for (RankedTensorType type : {lhs, rhs, result})
    if (type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0)
      return emitOpError() << "requires positive dimensions, got " << type;

  if (lhs.getDimSize(1) != rhs.getDimSize(0))
    return emitOpError() << "requires matching K dimensions, got lhs K = "
                         << lhs.getDimSize(1) << " and rhs K = "
                         << rhs.getDimSize(0);

  auto rows = lhs.getDimSize(0);
  auto columns = rhs.getDimSize(1);
  if (result.getDimSize(0) != rows || result.getDimSize(1) != columns)
    return emitOpError() << "requires result shape [" << rows << ", " << columns
                         << "], got " << result;
  return success();
}

#define GET_OP_CLASSES
#include "NPU/NPUOps.cpp.inc"
