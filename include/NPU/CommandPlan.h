#ifndef NPU_COMMANDPLAN_H
#define NPU_COMMANDPLAN_H

#include "NPU/DRAMLayout.h"
#include "NPU/MemoryPlan.h"
#include <array>
#include <variant>
#include <vector>

namespace mlir::npu {

struct ZeroCommand {
  int64_t bufferId, dstSram, rows, cols;
};

struct DMALoad2DCommand {
  int64_t tensorId, bufferId, srcDram, dstSram, rows, rowBytes, srcStrideBytes,
      dstStrideBytes;
};

struct MatMulTileCommand {
  int64_t aBufferId, bBufferId, cBufferId, aSram, bSram, cSram, m, n, k;
  bool accumulate;
};

struct DMAStore2DCommand {
  int64_t tensorId, bufferId, srcSram, dstDram, rows, rowBytes, srcStrideBytes,
      dstStrideBytes;
};

using CommandPayload =
    std::variant<ZeroCommand, DMALoad2DCommand, MatMulTileCommand,
                 DMAStore2DCommand>;

struct CommandRecord {
  int64_t eventIndex, tileIndex;
  CommandPayload payload;
};

using MatMulTensors = std::array<DRAMTensorDesc, 3>;
inline constexpr int64_t maxCommandRecords = 5 * maxTileRecords;

llvm::Expected<int64_t> checkedDMAEnd(int64_t base, int64_t rows,
                                     int64_t rowBytes, int64_t strideBytes);

llvm::Expected<std::vector<CommandRecord>>
generateCommands(llvm::ArrayRef<TileDesc> tiles, const MemoryPlan &memory,
                 const MatMulTensors &tensors);

llvm::Error verifyCommands(llvm::ArrayRef<TileDesc> tiles,
                           const MemoryPlan &memory,
                           const MatMulTensors &tensors,
                           llvm::ArrayRef<CommandRecord> commands);

} // namespace mlir::npu

#endif
