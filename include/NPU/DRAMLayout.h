#ifndef NPU_DRAMLAYOUT_H
#define NPU_DRAMLAYOUT_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
class ModuleOp;

namespace npu {

enum class TensorStorageKind { Input, Constant, Intermediate, Output };

struct DRAMTensorDesc {
  int64_t id;
  TensorStorageKind kind;
  int64_t sourceIndex, base, rows, cols, rowStrideBytes, sizeBytes;
};

struct DRAMLayout {
  int64_t alignmentBytes, totalBytes;
  std::vector<DRAMTensorDesc> tensors;
  std::vector<int64_t> inputIds, constantIds, outputIds;
};

inline constexpr char dramLayoutAttr[] = "npu.dram_layout";
inline constexpr int64_t dramTensorAlignment = 64;
inline constexpr std::size_t maxDRAMRecords = 65536;

enum class DRAMLayoutStage { SRAMPlanned, Commands };

llvm::Expected<DRAMLayout>
buildDRAMLayout(ModuleOp module, DRAMLayoutStage stage,
                llvm::DenseMap<Value, int64_t> &valueIds);

}
}

#endif
