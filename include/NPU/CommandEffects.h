#ifndef NPU_COMMANDEFFECTS_H
#define NPU_COMMANDEFFECTS_H

#include "NPU/CommandPlan.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::npu {

enum class MemorySpace { DRAM, SRAM };

struct ByteRange {
  int64_t begin, end;

  bool overlaps(const ByteRange &other) const {
    return begin < other.end && other.begin < end;
  }
};

struct MemoryAccess {
  MemorySpace space;
  ByteRange range;
};

struct CommandEffects {
  llvm::SmallVector<MemoryAccess, 3> reads;
  llvm::SmallVector<MemoryAccess, 1> writes;
};

llvm::Expected<CommandEffects> getCommandEffects(const CommandRecord &command);

}

#endif
