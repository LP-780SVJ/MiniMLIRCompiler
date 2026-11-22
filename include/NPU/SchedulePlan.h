#ifndef NPU_SCHEDULEPLAN_H
#define NPU_SCHEDULEPLAN_H

#include "NPU/CommandPlan.h"

namespace mlir::npu {

struct CommandKey {
  int64_t matmulIndex, localEventIndex;
};

struct ScheduledCommand {
  int64_t commandId;
  CommandKey origin;
  int64_t sequenceIndex;
  std::vector<int64_t> dependencies;
};

struct ExecutionSchedule {
  std::vector<ScheduledCommand> commands;
};

struct ScheduleMatMul {
  std::array<int64_t, 3> tensorIds;
  int64_t reductionSize;
  std::vector<TileDesc> tiles;
  MemoryPlan memory;
  std::vector<CommandRecord> commands;
};

inline constexpr int64_t maxScheduleCommands = maxCommandRecords;
inline constexpr int64_t maxScheduleEdges = 16 * maxScheduleCommands;

llvm::Expected<std::vector<std::vector<int64_t>>>
buildDependencies(const DRAMLayout &layout,
                  llvm::ArrayRef<ScheduleMatMul> matmuls);

llvm::Error verifyScheduleState(const DRAMLayout &layout,
                                llvm::ArrayRef<ScheduleMatMul> matmuls,
                                llvm::ArrayRef<ScheduledCommand> ordered);

llvm::Expected<ExecutionSchedule>
buildSchedule(const DRAMLayout &layout, llvm::ArrayRef<ScheduleMatMul> matmuls);

llvm::Error verifySchedule(const DRAMLayout &layout,
                           llvm::ArrayRef<ScheduleMatMul> matmuls,
                           const ExecutionSchedule &schedule);

}

#endif
