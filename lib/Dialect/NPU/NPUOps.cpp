#include "NPU/NPUOps.h"
#include "NPU/CommandPlanAttrs.h"
#include "NPU/MemoryPlanAttrs.h"
#include "NPU/TilePlanAttrs.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

#include <utility>

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

template <typename MatMul>
LogicalResult verifyTilePlan(MatMul op,
                             SmallVectorImpl<TileDesc> *decoded = nullptr) {
  auto lhs = op.getLhs().getType();
  auto rhs = op.getRhs().getType();
  if (failed(verifyMatMulTypes(op.getOperation(), lhs, rhs,
                               op.getResult().getType())))
    return failure();
  if (op.getTarget() != virtualNPUTarget)
    return op.emitOpError() << "unsupported tile plan target " << op.getTarget();

  auto plan = TilePlan::create(
      {lhs.getDimSize(0), rhs.getDimSize(1), lhs.getDimSize(1)},
      {op.getTileMAttr().getInt(), op.getTileNAttr().getInt(),
       op.getTileKAttr().getInt()});
  if (!plan)
    return op.emitOpError() << llvm::toString(plan.takeError());
  if (op.getTiles().size() != static_cast<size_t>(plan->size()))
    return op.emitOpError() << "expected " << plan->size() << " tile records, got "
                            << op.getTiles().size();

  for (auto [index, entry] : llvm::enumerate(op.getTiles())) {
    auto dict = dyn_cast<DictionaryAttr>(entry);
    if (!dict)
      return op.emitOpError() << "tile #" << index << " must be a dictionary";
    auto tile = decodeTileDesc(dict);
    if (!tile)
      return op.emitOpError() << "tile #" << index << ": "
                              << llvm::toString(tile.takeError());
    TileDesc expected = plan->getTile(index);
    if (tile->mOffset != expected.mOffset || tile->nOffset != expected.nOffset ||
        tile->kOffset != expected.kOffset || tile->mSize != expected.mSize ||
        tile->nSize != expected.nSize || tile->kSize != expected.kSize)
      return op.emitOpError() << "tile #" << index << " must have offset ("
                              << expected.mOffset << ", " << expected.nOffset
                              << ", " << expected.kOffset << ") and size ("
                              << expected.mSize << ", " << expected.nSize
                              << ", " << expected.kSize
                              << ") for full M/N/K coverage";
    if (tile->outputTileId != expected.outputTileId)
      return op.emitOpError() << "tile #" << index << " requires output_tile_id "
                              << expected.outputTileId;
    if (tile->isFirstK != expected.isFirstK || tile->isLastK != expected.isLastK)
      return op.emitOpError() << "tile #" << index
                              << " has incorrect is_first_k / is_last_k flags";
    if (decoded)
      decoded->push_back(*tile);
  }
  return success();
}

}

LogicalResult MatMulOp::verify() {
  return verifyMatMulTypes(getOperation(), getLhs().getType(),
                           getRhs().getType(), getResult().getType());
}

LogicalResult TiledMatMulOp::verify() { return verifyTilePlan(*this); }

LogicalResult SRAMPlannedMatMulOp::verify() {
  SmallVector<TileDesc> tiles;
  if (failed(verifyTilePlan(*this, &tiles)))
    return failure();
  auto plan = decodeMemoryPlan(getMemoryPlan());
  if (!plan)
    return emitOpError() << llvm::toString(plan.takeError());
  if (auto error = verifyMemoryPlan(tiles, *plan))
    return emitOpError() << llvm::toString(std::move(error));
  return success();
}

LogicalResult CommandMatMulOp::verify() {
  SmallVector<TileDesc> tiles;
  if (failed(verifyTilePlan(*this, &tiles)))
    return failure();
  auto memory = decodeMemoryPlan(getMemoryPlan());
  if (!memory)
    return emitOpError() << llvm::toString(memory.takeError());
  if (auto error = verifyMemoryPlan(tiles, *memory))
    return emitOpError() << llvm::toString(std::move(error));
  auto module = getOperation()->getParentOfType<ModuleOp>();
  auto layout = module ? module->getAttrOfType<DictionaryAttr>(dramLayoutAttr)
                       : DictionaryAttr{};
  if (!layout)
    return emitOpError("requires a module DRAM layout (npu.dram_layout)");
  auto descriptors = dyn_cast_or_null<ArrayAttr>(layout.get("tensors"));
  if (!descriptors)
    return emitOpError("requires a DRAM tensor descriptor array");

  MatMulTensors tensors;
  const int64_t ids[] = {getLhsTensorIdAttr().getInt(),
                         getRhsTensorIdAttr().getInt(),
                         getResultTensorIdAttr().getInt()};
  for (auto [index, id] : llvm::enumerate(ids)) {
    if (id < 0 || static_cast<uint64_t>(id) >= descriptors.size())
      return emitOpError() << "unknown DRAM tensor ID " << id;
    auto dict = dyn_cast<DictionaryAttr>(descriptors[id]);
    if (!dict)
      return emitOpError("DRAM tensor descriptor must be a dictionary");
    auto desc = decodeDRAMTensorDesc(dict);
    if (!desc)
      return emitOpError() << llvm::toString(desc.takeError());
    tensors[index] = *desc;
  }
  auto commands = decodeCommands(getCommands());
  if (!commands)
    return emitOpError() << llvm::toString(commands.takeError());
  if (auto error = verifyCommands(tiles, *memory, tensors, *commands))
    return emitOpError() << llvm::toString(std::move(error));
  return success();
}

#define GET_OP_CLASSES
#include "NPU/NPUOps.cpp.inc"
