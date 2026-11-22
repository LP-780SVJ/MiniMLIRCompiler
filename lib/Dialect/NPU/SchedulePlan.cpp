#include "NPU/SchedulePlan.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <utility>

using namespace mlir::npu;

namespace {

llvm::Expected<std::vector<int64_t>>
getOffsets(llvm::ArrayRef<ScheduleMatMul> matmuls) {
  if (matmuls.empty())
    return llvm::createStringError("schedule requires at least one command MatMul");
  std::vector<int64_t> offsets{0};
  for (const auto &matmul : matmuls) {
    if (matmul.commands.empty() ||
        matmul.commands.size() >
            static_cast<size_t>(maxScheduleCommands - offsets.back()))
      return llvm::createStringError("empty command group or schedule exceeds command limit");
    offsets.push_back(offsets.back() + matmul.commands.size());
  }
  return offsets;
}

llvm::Error checkGraph(llvm::ArrayRef<const ScheduledCommand *> byId) {
  std::vector<std::vector<int64_t>> successors(byId.size());
  std::vector<size_t> pending(byId.size());
  std::vector<int64_t> ready;
  size_t edgeCount = 0;
  for (auto [id, command] : llvm::enumerate(byId)) {
    const auto &deps = command->dependencies;
    if (deps.size() > maxScheduleEdges - edgeCount)
      return llvm::createStringError("schedule exceeds dependency edge limit");
    edgeCount += deps.size();
    int64_t last = -1;
    for (int64_t dep : deps) {
      if (dep < 0 || static_cast<uint64_t>(dep) >= byId.size())
        return llvm::createStringError("dependency refers to an unknown command ID");
      if (dep <= last)
        return llvm::createStringError("dependencies must be sorted and unique");
      last = dep;
      successors[dep].push_back(id);
    }
    pending[id] = deps.size();
    if (deps.empty())
      ready.push_back(id);
  }
  for (size_t head = 0; head < ready.size(); ++head)
    for (int64_t next : successors[ready[head]])
      if (--pending[next] == 0)
        ready.push_back(next);
  if (ready.size() != byId.size())
    return llvm::createStringError("cycle in command dependencies");
  return llvm::Error::success();
}

}

llvm::Expected<ExecutionSchedule>
mlir::npu::buildSchedule(const DRAMLayout &layout,
                         llvm::ArrayRef<ScheduleMatMul> matmuls) {
  auto offsets = getOffsets(matmuls);
  if (!offsets)
    return offsets.takeError();
  auto deps = buildDependencies(layout, matmuls);
  if (!deps)
    return deps.takeError();

  ExecutionSchedule schedule;
  schedule.commands.reserve(offsets->back());
  for (auto [matmulIndex, matmul] : llvm::enumerate(matmuls)) {
    for (auto [localIndex, command] : llvm::enumerate(matmul.commands)) {
      if (command.eventIndex != static_cast<int64_t>(localIndex))
        return llvm::createStringError("noncanonical local event index");
      int64_t id = (*offsets)[matmulIndex] + localIndex;
      schedule.commands.push_back(
          {id, {static_cast<int64_t>(matmulIndex), command.eventIndex},
           id, std::move((*deps)[id])});
    }
  }
  return schedule;
}

llvm::Error mlir::npu::verifySchedule(const DRAMLayout &layout,
                                      llvm::ArrayRef<ScheduleMatMul> matmuls,
                                      const ExecutionSchedule &schedule) {
  auto offsets = getOffsets(matmuls);
  if (!offsets)
    return offsets.takeError();
  size_t count = offsets->back();
  if (schedule.commands.size() != count)
    return llvm::createStringError("schedule must cover every command exactly once");

  std::vector<const ScheduledCommand *> byId(count, nullptr);
  std::vector<ScheduledCommand> ordered(count);
  std::vector<bool> usedSequence(count);
  for (const auto &command : schedule.commands) {
    auto [matmulIndex, localIndex] = command.origin;
    if (matmulIndex < 0 || static_cast<uint64_t>(matmulIndex) >= matmuls.size())
      return llvm::createStringError("unknown MatMul index in schedule");
    if (localIndex < 0 ||
        static_cast<uint64_t>(localIndex) >= matmuls[matmulIndex].commands.size() ||
        matmuls[matmulIndex].commands[localIndex].eventIndex != localIndex)
      return llvm::createStringError("unknown local event index in schedule");
    int64_t id = command.commandId;
    if (id < 0 || static_cast<uint64_t>(id) >= count)
      return llvm::createStringError("invalid global command ID");
    if (byId[id])
      return llvm::createStringError("duplicate global command ID");
    if (id != (*offsets)[matmulIndex] + localIndex)
      return llvm::createStringError("global command ID does not match MatMul and local event");
    byId[id] = &command;

    int64_t sequence = command.sequenceIndex;
    if (sequence < 0 || static_cast<uint64_t>(sequence) >= count)
      return llvm::createStringError("sequence index is outside the contiguous schedule");
    if (usedSequence[sequence])
      return llvm::createStringError("duplicate sequence index");
    usedSequence[sequence] = true;
    ordered[sequence] = command;
  }
  if (auto error = checkGraph(byId))
    return error;
  if (auto error = verifyScheduleState(layout, matmuls, ordered))
    return error;

  auto required = buildDependencies(layout, matmuls);
  if (!required)
    return required.takeError();
  for (auto [id, command] : llvm::enumerate(byId)) {
    const auto &deps = command->dependencies;
    const auto &expected = (*required)[id];
    if (!std::includes(deps.begin(), deps.end(), expected.begin(), expected.end()))
      return llvm::createStringError(
          llvm::Twine("missing required dependency for command ") + llvm::Twine(id));
    for (int64_t dep : deps)
      if (byId[dep]->sequenceIndex >= command->sequenceIndex)
        return llvm::createStringError("dependency source must execute before its target");
    if (command->sequenceIndex != static_cast<int64_t>(id))
      return llvm::createStringError("sequence violates serial-mnk-single-buffer-v1 order");
  }
  return llvm::Error::success();
}
