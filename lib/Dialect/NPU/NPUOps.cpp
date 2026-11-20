#include "NPU/NPUOps.h"
#include "NPU/TilePlanAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::npu;

namespace {

LogicalResult verifyMatMulTypes(Operation *op, RankedTensorType lhs,
                                RankedTensorType rhs, RankedTensorType result) {
  if (lhs.getRank() != 2 || rhs.getRank() != 2 || result.getRank() != 2)
    return op->emitOpError() << "requires rank-2 tensors";

  for (RankedTensorType type : {lhs, rhs, result})
    if (type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0)
      return op->emitOpError() << "requires positive dimensions, got " << type;

  if (lhs.getDimSize(1) != rhs.getDimSize(0))
    return op->emitOpError()
           << "requires matching K dimensions, got lhs K = " << lhs.getDimSize(1)
           << " and rhs K = " << rhs.getDimSize(0);

  auto rows = lhs.getDimSize(0);
  auto columns = rhs.getDimSize(1);
  if (result.getDimSize(0) != rows || result.getDimSize(1) != columns)
    return op->emitOpError() << "requires result shape [" << rows << ", "
                             << columns << "], got " << result;
  return success();
}

}

LogicalResult MatMulOp::verify() {
  return verifyMatMulTypes(getOperation(), getLhs().getType(),
                           getRhs().getType(), getResult().getType());
}

LogicalResult TiledMatMulOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  if (failed(verifyMatMulTypes(getOperation(), lhs, rhs, getResult().getType())))
    return failure();
  if (getTarget() != virtualNPUTarget)
    return emitOpError() << "unsupported tile plan target " << getTarget();

  auto plan = TilePlan::create(
      {lhs.getDimSize(0), rhs.getDimSize(1), lhs.getDimSize(1)},
      {getTileMAttr().getInt(), getTileNAttr().getInt(), getTileKAttr().getInt()});
  if (!plan)
    return emitOpError() << llvm::toString(plan.takeError());
  if (getTiles().size() != static_cast<size_t>(plan->size()))
    return emitOpError() << "expected " << plan->size() << " tile records, got "
                         << getTiles().size();

  for (auto [index, entry] : llvm::enumerate(getTiles())) {
    auto dict = dyn_cast<DictionaryAttr>(entry);
    if (!dict)
      return emitOpError() << "tile #" << index << " must be a dictionary";
    auto tile = decodeTileDesc(dict);
    if (!tile)
      return emitOpError() << "tile #" << index << ": "
                           << llvm::toString(tile.takeError());
    TileDesc expected = plan->getTile(index);
    if (tile->mOffset != expected.mOffset || tile->nOffset != expected.nOffset ||
        tile->kOffset != expected.kOffset || tile->mSize != expected.mSize ||
        tile->nSize != expected.nSize || tile->kSize != expected.kSize)
      return emitOpError() << "tile #" << index << " must have offset ("
                           << expected.mOffset << ", " << expected.nOffset << ", "
                           << expected.kOffset << ") and size (" << expected.mSize
                           << ", " << expected.nSize << ", " << expected.kSize
                           << ") for full M/N/K coverage";
    if (tile->outputTileId != expected.outputTileId)
      return emitOpError() << "tile #" << index << " requires output_tile_id "
                           << expected.outputTileId;
    if (tile->isFirstK != expected.isFirstK || tile->isLastK != expected.isLastK)
      return emitOpError() << "tile #" << index
                           << " has incorrect is_first_k / is_last_k flags";
  }
  return success();
}

#define GET_OP_CLASSES
#include "NPU/NPUOps.cpp.inc"
