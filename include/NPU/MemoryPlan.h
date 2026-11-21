#ifndef NPU_MEMORYPLAN_H
#define NPU_MEMORYPLAN_H

#include "NPU/TilePlan.h"
#include "llvm/ADT/ArrayRef.h"
#include <vector>

namespace mlir::npu {

inline constexpr char serialSRAMSchedule[] = "serial-mnk-single-buffer-v1";

struct SRAMConfig {
  int64_t capacityBytes;
  int64_t reservedBytes;
  int64_t alignmentBytes;
};

enum class BufferRole { A, B, C };

struct BufferRequirement {
  int64_t id;
  BufferRole role;
  int64_t tileIndex;
  int64_t outputTileId;
  int64_t rawBytes;
  int64_t allocatedBytes;
  int64_t alignmentBytes;
  int64_t liveBegin;
  int64_t liveEnd;
};

struct BufferAllocation {
  BufferRequirement requirement;
  int64_t address;
  bool reusedAddress;
};

struct TileBufferBinding {
  int64_t tileIndex;
  int64_t aBufferId;
  int64_t bBufferId;
  int64_t cBufferId;
};

struct MemoryPlan {
  SRAMConfig config;
  std::vector<BufferAllocation> allocations;
  std::vector<TileBufferBinding> bindings;
  int64_t peakLiveBytes;
  int64_t arenaHighWaterBytes;
};

llvm::Error verifySRAMConfig(SRAMConfig config);
llvm::Expected<std::vector<BufferAllocation>>
allocateBuffers(llvm::ArrayRef<BufferRequirement> requirements, SRAMConfig config);
llvm::Expected<MemoryPlan> generateMemoryPlan(llvm::ArrayRef<TileDesc> tiles,
                                              SRAMConfig config);
llvm::Error verifyMemoryPlan(llvm::ArrayRef<TileDesc> tiles,
                             const MemoryPlan &plan);

}

#endif
