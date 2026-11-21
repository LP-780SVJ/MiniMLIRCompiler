#include "NPU/DRAMLayout.h"

#include "NPU/NPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::npu;

llvm::Expected<DRAMLayout>
mlir::npu::buildDRAMLayout(ModuleOp module, DRAMLayoutStage stage,
                          llvm::DenseMap<Value, int64_t> &valueIds) {
  if (module->getNumRegions() != 1 ||
      !llvm::hasSingleElement(module->getRegion(0)) ||
      !llvm::hasSingleElement(module.getBody()->getOperations()))
    return llvm::createStringError("expected a single defined @main function");
  auto entry = dyn_cast<func::FuncOp>(module.getBody()->front());
  auto name = entry ? entry->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
                    : StringAttr{};
  if (!name || name.getValue() != "main" || entry->getNumRegions() != 1 ||
      !llvm::hasSingleElement(entry.getBody()))
    return llvm::createStringError("expected a single defined @main function with one block");

  auto &block = entry.getBody().front();
  if (block.empty())
    return llvm::createStringError("expected at least one MatMul and a function return");
  SmallVector<Value> constants, results;
  func::ReturnOp returned;
  for (Operation &op : block) {
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (op.getNumResults() != 1 ||
          !isa_and_nonnull<DenseFPElementsAttr>(op.getAttr("value")))
        return llvm::createStringError("expected a Dense FP32 Tensor constant");
      constants.push_back(op.getResult(0));
    } else if ((stage == DRAMLayoutStage::SRAMPlanned &&
                isa<SRAMPlannedMatMulOp>(op)) ||
               (stage == DRAMLayoutStage::Commands && isa<CommandMatMulOp>(op))) {
      if (op.getNumOperands() != 2 || op.getNumResults() != 1)
        return llvm::createStringError("invalid MatMul operand or result count");
      results.push_back(op.getResult(0));
    } else if (auto ret = dyn_cast<func::ReturnOp>(op); ret && &op == &block.back()) {
      returned = ret;
    } else {
      return llvm::createStringError(
          llvm::Twine("unsupported DMA input operation '") + op.getName().getStringRef() +
          "'; run SRAM planning before DMA insertion");
    }
  }
  if (!returned || results.empty())
    return llvm::createStringError("expected at least one MatMul and a function return");
  if (returned->getNumOperands() > maxDRAMRecords)
    return llvm::createStringError("too many DRAM output records");

  DRAMLayout layout{dramTensorAlignment, 0, {}, {}, {}, {}};
  valueIds.clear();
  auto appendTensor = [&](Value value, TensorStorageKind kind,
                          int64_t ordinal) -> llvm::Expected<int64_t> {
    if (layout.tensors.size() >= maxDRAMRecords)
      return llvm::createStringError("too many DRAM tensor records");
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getRank() != 2 || !type.getElementType().isF32() ||
        type.getEncoding() || type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0)
      return llvm::createStringError(
          "DRAM layout requires positive static rank-2 FP32 tensors without encoding");
    int64_t stride, size, base, end;
    int64_t remainder = layout.totalBytes % layout.alignmentBytes;
    int64_t padding = remainder ? layout.alignmentBytes - remainder : 0;
    if (llvm::MulOverflow(type.getDimSize(1), int64_t{4}, stride) ||
        llvm::MulOverflow(type.getDimSize(0), stride, size) ||
        llvm::AddOverflow(layout.totalBytes, padding, base) ||
        llvm::AddOverflow(base, size, end))
      return llvm::createStringError("DRAM tensor address or byte size overflow");
    int64_t id = static_cast<int64_t>(layout.tensors.size());
    valueIds.try_emplace(value, id);
    layout.tensors.push_back({id, kind, ordinal, base, type.getDimSize(0),
                              type.getDimSize(1), stride, size});
    layout.totalBytes = end;
    return id;
  };

  for (auto [index, arg] : llvm::enumerate(block.getArguments())) {
    auto id = appendTensor(arg, TensorStorageKind::Input, index);
    if (!id)
      return id.takeError();
    layout.inputIds.push_back(*id);
  }
  for (auto [index, constant] : llvm::enumerate(constants)) {
    auto id = appendTensor(constant, TensorStorageKind::Constant, index);
    if (!id)
      return id.takeError();
    layout.constantIds.push_back(*id);
  }
  llvm::DenseSet<Value> outputs(returned.getOperands().begin(),
                                returned.getOperands().end());
  for (auto [index, result] : llvm::enumerate(results)) {
    auto id = appendTensor(result, outputs.contains(result) ? TensorStorageKind::Output
                                                            : TensorStorageKind::Intermediate,
                           index);
    if (!id)
      return id.takeError();
  }
  for (Value output : returned.getOperands()) {
    auto id = valueIds.find(output);
    if (id == valueIds.end())
      return llvm::createStringError("unmapped function return value");
    layout.outputIds.push_back(id->second);
  }
  return layout;
}
