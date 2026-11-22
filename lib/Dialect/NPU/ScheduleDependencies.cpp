#include "NPU/SchedulePlan.h"

#include "NPU/CommandEffects.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
#include <iterator>
#include <map>
#include <utility>

using namespace mlir::npu;

namespace {

class IncomingEdges {
public:
  IncomingEdges(int64_t commandId, size_t &edgeCount)
      : commandId(commandId), edgeCount(edgeCount) {}

  llvm::Error add(int64_t source) {
    if (source < 0 || source == commandId || sources.contains(source))
      return llvm::Error::success();
    if (edgeCount == maxScheduleEdges)
      return llvm::createStringError("schedule exceeds dependency edge limit");
    ++edgeCount;
    sources.insert(source);
    return llvm::Error::success();
  }

  std::vector<int64_t> finish() const {
    std::vector<int64_t> result(sources.begin(), sources.end());
    std::sort(result.begin(), result.end());
    return result;
  }

private:
  int64_t commandId;
  size_t &edgeCount;
  llvm::SmallDenseSet<int64_t, 8> sources;
};

struct RangeHistory {
  int64_t end;
  int64_t writer;
  llvm::SmallVector<int64_t, 2> readers;
};

class SRAMHazards {
public:
  llvm::Error read(ByteRange range, int64_t id, IncomingEdges &incoming) {
    auto current = splitAt(range.begin);
    auto past = splitAt(range.end);
    for (int64_t begin = range.begin; begin < range.end;) {
      if (current == past || current->first > begin) {
        int64_t end = current == past ? range.end : current->first;
        ranges.emplace_hint(current, begin, RangeHistory{end, -1, {id}});
        begin = end;
        continue;
      }
      auto &history = current->second;
      if (auto error = incoming.add(history.writer))
        return error;
      if (history.readers.empty() || history.readers.back() != id)
        history.readers.push_back(id);
      begin = history.end;
      ++current;
    }
    return llvm::Error::success();
  }

  llvm::Error write(ByteRange range, int64_t id, IncomingEdges &incoming) {
    auto first = splitAt(range.begin);
    auto past = splitAt(range.end);
    for (auto current = first; current != past; ++current) {
      if (auto error = incoming.add(current->second.writer))
        return error;
      for (int64_t reader : current->second.readers)
        if (auto error = incoming.add(reader))
          return error;
    }
    auto next = ranges.erase(first, past);
    ranges.emplace_hint(next, range.begin, RangeHistory{range.end, id, {}});
    return llvm::Error::success();
  }

private:
  using RangeMap = std::map<int64_t, RangeHistory>;
  RangeMap::iterator splitAt(int64_t boundary) {
    auto next = ranges.lower_bound(boundary);
    if (next == ranges.begin() ||
        (next != ranges.end() && next->first == boundary))
      return next;
    auto previous = std::prev(next);
    if (previous->second.end <= boundary)
      return next;
    auto right = previous->second;
    previous->second.end = boundary;
    return ranges.emplace_hint(next, boundary, std::move(right));
  }

  RangeMap ranges;
};

}

llvm::Expected<std::vector<std::vector<int64_t>>>
mlir::npu::buildDependencies(const DRAMLayout &layout,
                            llvm::ArrayRef<ScheduleMatMul> matmuls) {
  llvm::DenseMap<int64_t, TensorStorageKind> tensorKinds;
  for (const auto &tensor : layout.tensors)
    if (tensor.id < 0 || !tensorKinds.try_emplace(tensor.id, tensor.kind).second)
      return llvm::createStringError("invalid or duplicate DRAM tensor ID in dependency analysis");

  std::vector<std::vector<int64_t>> graph;
  size_t edgeCount = 0;
  llvm::DenseMap<int64_t, int64_t> tensorReady;
  SRAMHazards sram;
  for (const ScheduleMatMul &matmul : matmuls) {
    if (matmul.commands.empty() ||
        matmul.commands.size() > maxScheduleCommands - graph.size())
      return llvm::createStringError("empty command group or schedule exceeds command limit");

    int64_t previousMatMulEnd = static_cast<int64_t>(graph.size()) - 1;
    int64_t lastStore = -1;
    int64_t outputTile = -1;
    int64_t previousCompute = -1;
    for (const CommandRecord &command : matmul.commands) {
      int64_t id = static_cast<int64_t>(graph.size());
      IncomingEdges incoming(id, edgeCount);
      if (id == previousMatMulEnd + 1)
        if (auto error = incoming.add(previousMatMulEnd))
          return error;

      auto effects = getCommandEffects(command);
      if (!effects)
        return effects.takeError();
      for (const auto &access : effects->reads)
        if (access.space == MemorySpace::SRAM)
          if (auto error = sram.read(access.range, id, incoming))
            return error;
      for (const auto &access : effects->writes)
        if (access.space == MemorySpace::SRAM)
          if (auto error = sram.write(access.range, id, incoming))
            return error;

      if (const auto *load = std::get_if<DMALoad2DCommand>(&command.payload)) {
        auto kind = tensorKinds.find(load->tensorId);
        if (kind == tensorKinds.end())
          return llvm::createStringError("LOAD references unknown DRAM tensor");
        if (kind->second == TensorStorageKind::Intermediate ||
            kind->second == TensorStorageKind::Output) {
          auto completed = tensorReady.find(load->tensorId);
          if (completed == tensorReady.end())
            return llvm::createStringError("LOAD precedes complete DRAM tensor writeback");
          if (auto error = incoming.add(completed->second))
            return error;
        }
      } else if (const auto *store =
                     std::get_if<DMAStore2DCommand>(&command.payload)) {
        auto kind = tensorKinds.find(store->tensorId);
        if (store->tensorId != matmul.tensorIds[2] ||
            kind == tensorKinds.end() ||
            (kind->second != TensorStorageKind::Intermediate &&
             kind->second != TensorStorageKind::Output))
          return llvm::createStringError("STORE does not write the MatMul result tensor");
        if (auto error = incoming.add(lastStore))
          return error;
        lastStore = id;
      } else if (std::holds_alternative<MatMulTileCommand>(command.payload)) {
        if (command.tileIndex < 0 ||
            static_cast<size_t>(command.tileIndex) >= matmul.tiles.size())
          return llvm::createStringError("unknown compute tile in dependency analysis");
        int64_t currentOutput = matmul.tiles[command.tileIndex].outputTileId;
        if (currentOutput == outputTile)
          if (auto error = incoming.add(previousCompute))
            return error;
        outputTile = currentOutput;
        previousCompute = id;
      }

      graph.push_back(incoming.finish());
    }
    if (lastStore < 0 ||
        !tensorReady.try_emplace(matmul.tensorIds[2], lastStore).second)
      return llvm::createStringError("missing or duplicate DRAM tensor producer");
  }
  return graph;
}
