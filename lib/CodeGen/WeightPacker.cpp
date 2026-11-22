#include "NPU/ProgramPackage.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Endian.h"

using namespace mlir;
using namespace mlir::npu;

llvm::Expected<PackedWeights>
mlir::npu::packWeights(ModuleOp module, const DRAMLayout &layout) {
  SmallVector<DenseFPElementsAttr> constants;
  for (Operation &op : module.getBody()->front().getRegion(0).front()) {
    if (auto constant = dyn_cast<arith::ConstantOp>(op))
      constants.push_back(dyn_cast<DenseFPElementsAttr>(constant.getValue()));
  }
  if (constants.size() != layout.constantIds.size())
    return llvm::createStringError("constant count does not match DRAM layout");

  PackedWeights result;
  for (auto [ordinal, id] : llvm::enumerate(layout.constantIds)) {
    if (id < 0 || static_cast<uint64_t>(id) >= layout.tensors.size())
      return llvm::createStringError("invalid constant tensor ID");
    const auto &tensor = layout.tensors[id];
    if (tensor.id != id || tensor.kind != TensorStorageKind::Constant ||
        tensor.sourceIndex != static_cast<int64_t>(ordinal))
      return llvm::createStringError("invalid constant tensor descriptor");
    auto attr = constants[ordinal];
    if (!attr || !attr.getElementType().isF32() ||
        attr.getType().getShape() != ArrayRef<int64_t>{tensor.rows, tensor.cols} ||
        tensor.sizeBytes <= 0 || attr.getNumElements() != tensor.sizeBytes / 4 ||
        tensor.sizeBytes % 4)
      return llvm::createStringError("constant payload does not match DRAM descriptor");

    int64_t offset = static_cast<int64_t>(result.bytes.size());
    int64_t padding = (dramTensorAlignment - offset % dramTensorAlignment) %
                      dramTensorAlignment;
    offset += padding;
    if (tensor.sizeBytes > maxJSONInteger - offset ||
        static_cast<uint64_t>(offset + tensor.sizeBytes) > result.bytes.max_size())
      return llvm::createStringError("packed weights exceed supported file size");
    result.bytes.resize(offset + tensor.sizeBytes, '\0');
    char *out = result.bytes.data() + offset;
    for (const APFloat &element : attr.getValues<APFloat>()) {
      llvm::support::endian::write32le(out, element.bitcastToAPInt().getZExtValue());
      out += 4;
    }
    result.entries.push_back(llvm::json::Object{{"tensor_id", id},
                                                {"file_offset", offset},
                                                {"size_bytes", tensor.sizeBytes},
                                                {"dram_base", tensor.base}});
  }
  return result;
}
