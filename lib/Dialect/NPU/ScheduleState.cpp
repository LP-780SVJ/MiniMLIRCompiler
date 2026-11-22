#include "NPU/SchedulePlan.h"

#include "NPU/CommandEffects.h"
#include "llvm/Support/MathExtras.h"

#include <iterator>
#include <map>
#include <unordered_map>
#include <utility>

using namespace mlir::npu;

namespace {

struct BufferState {
  const BufferAllocation *allocation;
  int64_t firstAccess = -1;
  int64_t lastAccess = -1;

  void access(int64_t rank) {
    if (firstAccess < 0)
      firstAccess = rank;
    lastAccess = rank;
  }
};

struct OutputState {
  bool initialized = false;
  bool stored = false;
  int64_t nextK = 0;
};

struct MatMulState {
  std::unordered_map<int64_t, BufferState> buffers;
  std::vector<OutputState> outputs;
  int64_t executed = 0;
};

struct TensorState {
  int64_t remainingStores = 0;
  bool available = false;
};

struct SRAMContent {
  ByteRange allocation;
  ByteRange written;
  const BufferState *owner;
};

const DRAMTensorDesc *findTensor(const DRAMLayout &layout, int64_t id) {
  if (id < 0 || static_cast<uint64_t>(id) >= layout.tensors.size() ||
      layout.tensors[id].id != id)
    return nullptr;
  return &layout.tensors[id];
}

llvm::Expected<size_t> countOutputRegions(llvm::ArrayRef<TileDesc> tiles,
                                           const DRAMTensorDesc &result) {
  int64_t nextRow = 0, nextCol = 0, bandRows = 0;
  size_t outputs = 0;
  const TileDesc *current = nullptr;
  for (const auto &tile : tiles) {
    if (tile.isFirstK) {
      if (current || tile.outputTileId != static_cast<int64_t>(outputs) ||
          tile.mOffset != nextRow || tile.nOffset != nextCol ||
          tile.mSize <= 0 || tile.nSize <= 0 || nextRow >= result.rows ||
          nextCol >= result.cols || tile.mSize > result.rows - nextRow ||
          tile.nSize > result.cols - nextCol ||
          (nextCol != 0 && tile.mSize != bandRows))
        return llvm::createStringError("STORE output tiles do not form a complete disjoint cover");
      bandRows = tile.mSize;
      nextCol += tile.nSize;
      if (nextCol == result.cols) {
        nextRow += bandRows;
        nextCol = 0;
      }
      ++outputs;
      current = &tile;
    }
    if (!current || tile.outputTileId != current->outputTileId ||
        tile.mOffset != current->mOffset || tile.nOffset != current->nOffset ||
        tile.mSize != current->mSize || tile.nSize != current->nSize)
      return llvm::createStringError("K segments do not use the same STORE output region");
    if (tile.isLastK)
      current = nullptr;
  }
  if (outputs == 0 || current || nextCol != 0 || nextRow != result.rows)
    return llvm::createStringError("STORE output tiles do not form a complete disjoint cover");
  return outputs;
}

llvm::Error verifyDRAMRegion(const DRAMTensorDesc &tensor, int64_t row,
                             int64_t col, int64_t rows, int64_t cols,
                             int64_t address, int64_t commandRows,
                             int64_t rowBytes, int64_t stride) {
  if (row < 0 || col < 0 || rows <= 0 || cols <= 0 || row >= tensor.rows ||
      col >= tensor.cols || rows > tensor.rows - row || cols > tensor.cols - col)
    return llvm::createStringError("command region is outside the bound DRAM Tensor");
  int64_t rowOffset, colOffset, offset, start, width;
  if (llvm::MulOverflow(row, tensor.rowStrideBytes, rowOffset) ||
      llvm::MulOverflow(col, int64_t{4}, colOffset) ||
      llvm::AddOverflow(rowOffset, colOffset, offset) ||
      llvm::AddOverflow(tensor.base, offset, start) ||
      llvm::MulOverflow(cols, int64_t{4}, width))
    return llvm::createStringError("DRAM region byte offset overflow");
  if (address != start || rows != commandRows || rowBytes != width ||
      stride != tensor.rowStrideBytes)
    return llvm::createStringError("command DRAM access does not match its Tile region");
  return llvm::Error::success();
}

class SRAMState {
public:
  llvm::Error write(BufferState &buffer, ByteRange range, int64_t rank,
                    SRAMConfig config) {
    const auto &alloc = *buffer.allocation;
    int64_t base = alloc.address;
    int64_t capacity = alloc.requirement.allocatedBytes;
    if (base < 0 || base < config.reservedBytes || base > config.capacityBytes ||
        capacity <= 0 || capacity > config.capacityBytes - base ||
        alloc.requirement.rawBytes <= 0 || alloc.requirement.rawBytes > capacity)
      return llvm::createStringError("invalid SRAM allocation range");
    ByteRange allocation{base, base + capacity};
    ByteRange usable{base, base + alloc.requirement.rawBytes};
    if (range.begin != usable.begin || range.end != usable.end)
      return llvm::createStringError("command write does not match its SRAM buffer's valid bytes");

    auto next = active.lower_bound(base);
    if ((next != active.end() && next->second.allocation.overlaps(allocation)) ||
        (next != active.begin() &&
         std::prev(next)->second.allocation.overlaps(allocation)))
      return llvm::createStringError("SRAM write overlaps a still-live buffer");
    active.emplace_hint(next, base, SRAMContent{allocation, range, &buffer});
    buffer.access(rank);
    return llvm::Error::success();
  }

  llvm::Error read(BufferState &buffer, ByteRange range, int64_t rank,
                   const char *diagnostic) const {
    auto found = active.find(buffer.allocation->address);
    if (found == active.end() || found->second.owner != &buffer ||
        !contains(found->second.written, range))
      return llvm::createStringError("%s", diagnostic);
    buffer.access(rank);
    return llvm::Error::success();
  }

  void release(const BufferState &buffer) {
    active.erase(buffer.allocation->address);
  }

  bool empty() const { return active.empty(); }

private:
  static bool contains(ByteRange bounds, ByteRange access) {
    return bounds.begin <= access.begin && access.begin < access.end &&
           access.end <= bounds.end;
  }

  std::map<int64_t, SRAMContent> active;
};

llvm::Error execute(const DRAMLayout &layout, const ScheduleMatMul &matmul,
                    const CommandRecord &command, MatMulState &state,
                    std::vector<TensorState> &dram, SRAMState &sram) {
  if (command.tileIndex < 0 ||
      static_cast<uint64_t>(command.tileIndex) >= matmul.tiles.size())
    return llvm::createStringError("unknown Tile for command");
  const auto &tile = matmul.tiles[command.tileIndex];
  if (tile.outputTileId < 0 ||
      static_cast<uint64_t>(tile.outputTileId) >= state.outputs.size())
    return llvm::createStringError("unknown output Tile");
  const auto &binding = matmul.memory.bindings[command.tileIndex];
  auto lookup = [&](int64_t id, BufferRole role) -> BufferState * {
    auto stateIt = state.buffers.find(id);
    if (stateIt == state.buffers.end())
      return nullptr;
    const auto &req = stateIt->second.allocation->requirement;
    if (req.role != role || req.outputTileId != tile.outputTileId ||
        (role != BufferRole::C && req.tileIndex != command.tileIndex))
      return nullptr;
    return &stateIt->second;
  };
  auto *bufferA = lookup(binding.aBufferId, BufferRole::A);
  auto *bufferB = lookup(binding.bBufferId, BufferRole::B);
  auto *bufferC = lookup(binding.cBufferId, BufferRole::C);
  if (!bufferA || !bufferB || !bufferC || binding.tileIndex != command.tileIndex)
    return llvm::createStringError("invalid SRAM Tile binding");
  auto &output = state.outputs[tile.outputTileId];
  auto effects = getCommandEffects(command);
  if (!effects)
    return effects.takeError();
  int64_t rank = state.executed++;
  if (const auto *zero = std::get_if<ZeroCommand>(&command.payload)) {
    if (output.initialized)
      return llvm::createStringError("repeated ZERO for the same output Tile");
    if (!tile.isFirstK || zero->bufferId != binding.cBufferId ||
        zero->rows != tile.mSize || zero->cols != tile.nSize)
      return llvm::createStringError("ZERO does not initialize the first K Tile's C buffer");
    if (auto error = sram.write(*bufferC, effects->writes[0].range, rank,
                                matmul.memory.config))
      return error;
    output.initialized = true;
    return llvm::Error::success();
  }

  if (const auto *load = std::get_if<DMALoad2DCommand>(&command.payload)) {
    bool isA = load->bufferId == binding.aBufferId;
    if (!isA && load->bufferId != binding.bBufferId)
      return llvm::createStringError("LOAD buffer does not match its A/B binding");
    const auto *tensor = findTensor(layout, load->tensorId);
    if (!tensor || load->tensorId != matmul.tensorIds[isA ? 0 : 1])
      return llvm::createStringError("LOAD tensor does not match its A/B operand");
    if (!dram[load->tensorId].available)
      return llvm::createStringError("LOAD reads a DRAM Tensor before all output tiles are written");
    if (load->dstStrideBytes != load->rowBytes)
      return llvm::createStringError("LOAD must pack SRAM rows without gaps");
    if (auto error = verifyDRAMRegion(
            *tensor, isA ? tile.mOffset : tile.kOffset,
            isA ? tile.kOffset : tile.nOffset,
            isA ? tile.mSize : tile.kSize, isA ? tile.kSize : tile.nSize,
            load->srcDram, load->rows, load->rowBytes, load->srcStrideBytes))
      return error;
    return sram.write(isA ? *bufferA : *bufferB, effects->writes[0].range, rank,
                      matmul.memory.config);
  }

  if (const auto *compute = std::get_if<MatMulTileCommand>(&command.payload)) {
    if (!output.initialized || output.stored)
      return llvm::createStringError("MATMUL reads C before ZERO or after STORE");
    if (tile.kOffset != output.nextK || tile.kSize <= 0 ||
        tile.kSize > matmul.reductionSize - output.nextK)
      return llvm::createStringError("MATMUL violates K reduction order");
    if (compute->aBufferId != binding.aBufferId ||
        compute->bBufferId != binding.bBufferId ||
        compute->cBufferId != binding.cBufferId || compute->m != tile.mSize ||
        compute->n != tile.nSize || compute->k != tile.kSize || !compute->accumulate)
      return llvm::createStringError("MATMUL does not match its Tile and SRAM binding");
    for (auto [buffer, index] :
         {std::pair{bufferA, 0}, std::pair{bufferB, 1}, std::pair{bufferC, 2}}) {
      if (auto error = sram.read(*buffer, effects->reads[index].range, rank,
                                 index == 2 ? "MATMUL reads invalid C contents"
                                            : "MATMUL reads A/B before LOAD or after overwrite"))
        return error;
    }
    sram.release(*bufferA);
    sram.release(*bufferB);
    output.nextK += tile.kSize;
    return llvm::Error::success();
  }

  const auto &store = std::get<DMAStore2DCommand>(command.payload);
  if (output.stored)
    return llvm::createStringError("duplicate STORE for the same output Tile");
  if (!output.initialized || !tile.isLastK || output.nextK != matmul.reductionSize)
    return llvm::createStringError("STORE before the K reduction is complete");
  const auto *tensor = findTensor(layout, store.tensorId);
  if (!tensor || store.tensorId != matmul.tensorIds[2] ||
      store.bufferId != binding.cBufferId)
    return llvm::createStringError("STORE does not match its C tensor and SRAM binding");
  if (store.srcStrideBytes != store.rowBytes)
    return llvm::createStringError("STORE must read compact SRAM rows");
  if (auto error = verifyDRAMRegion(*tensor, tile.mOffset, tile.nOffset, tile.mSize,
                                    tile.nSize, store.dstDram, store.rows,
                                    store.rowBytes, store.dstStrideBytes))
    return error;
  if (auto error = sram.read(*bufferC, effects->reads[0].range, rank,
                             "STORE reads invalid C contents"))
    return error;
  sram.release(*bufferC);
  output.stored = true;
  auto &result = dram[store.tensorId];
  result.available = --result.remainingStores == 0;
  return llvm::Error::success();
}

}

llvm::Error mlir::npu::verifyScheduleState(
    const DRAMLayout &layout, llvm::ArrayRef<ScheduleMatMul> matmuls,
    llvm::ArrayRef<ScheduledCommand> ordered) {
  std::vector<MatMulState> states(matmuls.size());
  std::vector<TensorState> dram(layout.tensors.size());
  for (size_t index = 0; index < layout.tensors.size(); ++index) {
    auto kind = layout.tensors[index].kind;
    dram[index].available =
        kind == TensorStorageKind::Input || kind == TensorStorageKind::Constant;
  }
  for (size_t index = 0; index < matmuls.size(); ++index) {
    const auto &matmul = matmuls[index];
    auto &state = states[index];
    int64_t resultId = matmul.tensorIds[2];
    const auto *result = findTensor(layout, resultId);
    if (!result || dram[resultId].available || dram[resultId].remainingStores != 0)
      return llvm::createStringError("invalid or duplicate MatMul result Tensor");
    if (matmul.reductionSize <= 0 || matmul.tiles.size() > maxTileRecords ||
        matmul.tiles.size() != matmul.memory.bindings.size())
      return llvm::createStringError("invalid Tile or SRAM Plan for scheduling");
    auto outputCount = countOutputRegions(matmul.tiles, *result);
    if (!outputCount)
      return outputCount.takeError();
    state.outputs.resize(*outputCount);
    dram[resultId].remainingStores = *outputCount;
    state.buffers.reserve(matmul.memory.allocations.size());
    for (const auto &alloc : matmul.memory.allocations)
      if (!state.buffers.emplace(alloc.requirement.id, BufferState{&alloc}).second)
        return llvm::createStringError("duplicate SRAM Buffer ID");
  }

  SRAMState sram;
  for (const auto &entry : ordered) {
    const auto &origin = entry.origin;
    if (origin.matmulIndex < 0 ||
        static_cast<uint64_t>(origin.matmulIndex) >= matmuls.size())
      return llvm::createStringError("unknown MatMul for schedule state");
    const auto &matmul = matmuls[origin.matmulIndex];
    if (origin.localEventIndex < 0 ||
        static_cast<uint64_t>(origin.localEventIndex) >= matmul.commands.size())
      return llvm::createStringError("unknown local event for schedule state");
    auto &state = states[origin.matmulIndex];
    if (auto error = execute(layout, matmul, matmul.commands[origin.localEventIndex],
                              state, dram, sram))
      return llvm::createStringError(
          llvm::Twine("sequence #") + llvm::Twine(entry.sequenceIndex) +
          " (MatMul #" + llvm::Twine(origin.matmulIndex) + ", event #" +
          llvm::Twine(origin.localEventIndex) + "): " +
          llvm::toString(std::move(error)));
  }

  for (size_t index = 0; index < matmuls.size(); ++index) {
    const auto &state = states[index];
    if (!dram[matmuls[index].tensorIds[2]].available)
      return llvm::createStringError("output Tensor has missing STORE regions");
    for (const auto &allocation : matmuls[index].memory.allocations) {
      const auto &requirement = allocation.requirement;
      const auto &buffer = state.buffers.at(requirement.id);
      if (buffer.firstAccess < 0 || buffer.firstAccess != requirement.liveBegin ||
          buffer.lastAccess + 1 != requirement.liveEnd)
        return llvm::createStringError(
            llvm::Twine("actual SRAM lifetime does not match the Memory Plan for MatMul #") +
            llvm::Twine(index) + " buffer #" + llvm::Twine(requirement.id));
    }
  }
  if (!sram.empty())
    return llvm::createStringError("SRAM buffer still live at end of schedule");
  for (int64_t id : layout.outputIds)
    if (!findTensor(layout, id) || !dram[id].available)
      return llvm::createStringError("function output Tensor is not initialized");
  return llvm::Error::success();
}
