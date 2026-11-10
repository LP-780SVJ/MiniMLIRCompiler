#include "Mini/MiniOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::mini;

namespace {

// ODS checks each type first; these rules check the relationship among types.
static LogicalResult verifyBinaryTensorTypes(Operation *op,
                                             RankedTensorType lhsType,
                                             RankedTensorType rhsType,
                                             RankedTensorType resultType) {
  if (lhsType.getShape() != rhsType.getShape())
    return op->emitOpError()
           << "requires lhs and rhs to have identical shapes, but got lhs "
           << lhsType << " and rhs " << rhsType;

  if (resultType.getShape() != lhsType.getShape())
    return op->emitOpError()
           << "requires result shape to match input shape, but got result "
           << resultType << " and input " << lhsType;

  // Equal dimensions alone would not catch a future difference in encoding.
  if (lhsType != rhsType || lhsType != resultType)
    return op->emitOpError()
           << "requires operand and result tensor types to be identical";

  return success();
}

} // namespace

LogicalResult AddOp::verify() {
  return verifyBinaryTensorTypes(
      getOperation(), llvm::cast<RankedTensorType>(getLhs().getType()),
      llvm::cast<RankedTensorType>(getRhs().getType()),
      llvm::cast<RankedTensorType>(getResult().getType()));
}

LogicalResult MulOp::verify() {
  return verifyBinaryTensorTypes(
      getOperation(), llvm::cast<RankedTensorType>(getLhs().getType()),
      llvm::cast<RankedTensorType>(getRhs().getType()),
      llvm::cast<RankedTensorType>(getResult().getType()));
}

LogicalResult ReluOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (inputType.getShape() != resultType.getShape())
    return emitOpError()
           << "requires result shape to match input shape, but got input "
           << inputType << " and result " << resultType;

  if (inputType != resultType)
    return emitOpError()
           << "requires result tensor type to be identical to input tensor type";

  return success();
}

#define GET_OP_CLASSES
#include "Mini/MiniOps.cpp.inc"
