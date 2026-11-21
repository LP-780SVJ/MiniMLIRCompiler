#include "NPU/CommandPlan.h"

#include "llvm/Support/MathExtras.h"
#include <unordered_map>
#include <utility>

using namespace mlir::npu;

llvm::Expected<int64_t>
mlir::npu::checkedDMAEnd(int64_t base, int64_t rows, int64_t rowBytes,
                        int64_t strideBytes) {
  if (base < 0 || rows <= 0 || rowBytes <= 0 || strideBytes < rowBytes)
    return llvm::createStringError("invalid 2D DMA dimensions or stride");
  int64_t lastRowOffset, lastRow, end;
  if (llvm::MulOverflow(rows - 1, strideBytes, lastRowOffset) ||
      llvm::AddOverflow(base, lastRowOffset, lastRow) ||
      llvm::AddOverflow(lastRow, rowBytes, end))
    return llvm::createStringError("DMA byte range overflow");
  return end;
}

namespace {

constexpr int64_t fp32Bytes = 4;

struct Window {
  int64_t start;
  int64_t rowBytes;
  int64_t stride;
};

llvm::Expected<Window> getDRAMWindow(const DRAMTensorDesc &tensor,
                                     int64_t row, int64_t col, int64_t rows,
                                     int64_t cols) {
  if (row < 0 || col < 0 || rows <= 0 || cols <= 0 || row >= tensor.rows ||
      col >= tensor.cols || rows > tensor.rows - row || cols > tensor.cols - col)
    return llvm::createStringError("tile region exceeds bound DRAM tensor shape");
  int64_t rowOffset, colOffset, offset, start, rowBytes;
  if (llvm::MulOverflow(row, tensor.rowStrideBytes, rowOffset) ||
      llvm::MulOverflow(col, fp32Bytes, colOffset) ||
      llvm::AddOverflow(rowOffset, colOffset, offset) ||
      llvm::AddOverflow(tensor.base, offset, start) ||
      llvm::MulOverflow(cols, fp32Bytes, rowBytes))
    return llvm::createStringError("DMA byte range overflow");
  return Window{start, rowBytes, tensor.rowStrideBytes};
}

llvm::Error verifyRange(int64_t start, int64_t rows, int64_t rowBytes,
                        int64_t stride, int64_t containerBase,
                        int64_t containerBytes, const char *diagnostic) {
  auto end = checkedDMAEnd(start, rows, rowBytes, stride);
  if (!end)
    return end.takeError();
  if (containerBase < 0 || containerBytes <= 0 || start < containerBase ||
      *end - containerBase > containerBytes)
    return llvm::createStringError("%s", diagnostic);
  return llvm::Error::success();
}

llvm::Error verifyDMA(const DRAMTensorDesc &tensor,
                      const BufferAllocation &buffer, int64_t dramStart,
                      int64_t sramStart, int64_t rows, int64_t rowBytes,
                      int64_t dramStride, int64_t sramStride) {
  if (auto error = verifyRange(dramStart, rows, rowBytes, dramStride, tensor.base,
                                tensor.sizeBytes,
                                "DMA access exceeds bound DRAM tensor"))
    return error;
  return verifyRange(sramStart, rows, rowBytes, sramStride, buffer.address,
                      buffer.requirement.rawBytes,
                      "DMA access exceeds bound SRAM buffer");
}

llvm::Error verifyCompactSRAM(const BufferAllocation &buffer, int64_t address,
                              int64_t rows, int64_t cols) {
  int64_t rowBytes;
  if (llvm::MulOverflow(cols, fp32Bytes, rowBytes))
    return llvm::createStringError("SRAM byte range overflow");
  return verifyRange(address, rows, rowBytes, rowBytes, buffer.address,
                      buffer.requirement.rawBytes,
                      "compute access exceeds bound SRAM buffer");
}

enum class EventKind { ZeroC, LoadA, LoadB, Compute, StoreC };

struct TileEvent {
  EventKind kind;
  int64_t index;
  int64_t tileIndex;
  const TileDesc &tile;
  const BufferAllocation &a;
  const BufferAllocation &b;
  const BufferAllocation &c;
};

size_t commandCount(llvm::ArrayRef<TileDesc> tiles) {
  size_t count = 3 * tiles.size();
  for (const TileDesc &tile : tiles)
    count += tile.isFirstK + tile.isLastK;
  return count;
}

template <typename Callback>
llvm::Error visitEvents(llvm::ArrayRef<TileDesc> tiles, const MemoryPlan &memory,
                        Callback callback) {
  if (memory.bindings.size() != tiles.size())
    return llvm::createStringError("tile binding count does not match tile plan");
  std::unordered_map<int64_t, const BufferAllocation *> buffers;
  buffers.reserve(memory.allocations.size());
  for (const BufferAllocation &alloc : memory.allocations)
    if (!buffers.emplace(alloc.requirement.id, &alloc).second)
      return llvm::createStringError("duplicate buffer ID");

  int64_t eventIndex = 0;
  for (size_t index = 0; index < tiles.size(); ++index) {
    const TileDesc &tile = tiles[index];
    const TileBufferBinding &binding = memory.bindings[index];
    auto a = buffers.find(binding.aBufferId);
    auto b = buffers.find(binding.bBufferId);
    auto c = buffers.find(binding.cBufferId);
    if (binding.tileIndex != static_cast<int64_t>(index) ||
        a == buffers.end() || b == buffers.end() || c == buffers.end())
      return llvm::createStringError("invalid tile buffer binding");
    auto emit = [&](EventKind kind) -> llvm::Error {
      TileEvent event{kind, eventIndex++, static_cast<int64_t>(index), tile,
                      *a->second, *b->second, *c->second};
      if (auto error = callback(event))
        return llvm::createStringError(llvm::Twine("event #") +
                                       llvm::Twine(event.index) + ": " +
                                       llvm::toString(std::move(error)));
      return llvm::Error::success();
    };
    if (tile.isFirstK)
      if (auto error = emit(EventKind::ZeroC))
        return error;
    if (auto error = emit(EventKind::LoadA))
      return error;
    if (auto error = emit(EventKind::LoadB))
      return error;
    if (auto error = emit(EventKind::Compute))
      return error;
    if (tile.isLastK)
      if (auto error = emit(EventKind::StoreC))
        return error;
  }
  return llvm::Error::success();
}

struct Transfer {
  const DRAMTensorDesc &tensor;
  const BufferAllocation &buffer;
  int64_t row, col, rows, cols;
};

Transfer getTransfer(const TileEvent &event, const MatMulTensors &tensors) {
  const TileDesc &tile = event.tile;
  switch (event.kind) {
  case EventKind::LoadA:
    return {tensors[0], event.a, tile.mOffset, tile.kOffset, tile.mSize,
            tile.kSize};
  case EventKind::LoadB:
    return {tensors[1], event.b, tile.kOffset, tile.nOffset, tile.kSize,
            tile.nSize};
  case EventKind::StoreC:
    return {tensors[2], event.c, tile.mOffset, tile.nOffset, tile.mSize,
            tile.nSize};
  default:
    llvm_unreachable("expected a transfer event");
  }
}

llvm::Expected<CommandPayload> buildPayload(const TileEvent &event,
                                             const MatMulTensors &tensors) {
  const TileDesc &tile = event.tile;
  if (event.kind == EventKind::ZeroC)
    return ZeroCommand{event.c.requirement.id, event.c.address, tile.mSize,
                       tile.nSize};
  if (event.kind == EventKind::Compute)
    return MatMulTileCommand{event.a.requirement.id,
                             event.b.requirement.id,
                             event.c.requirement.id,
                             event.a.address,
                             event.b.address,
                             event.c.address,
                             tile.mSize,
                             tile.nSize,
                             tile.kSize,
                             true};

  auto transfer = getTransfer(event, tensors);
  auto dram = getDRAMWindow(transfer.tensor, transfer.row, transfer.col,
                            transfer.rows, transfer.cols);
  if (!dram)
    return dram.takeError();
  if (event.kind == EventKind::StoreC)
    return DMAStore2DCommand{transfer.tensor.id, transfer.buffer.requirement.id,
                             transfer.buffer.address, dram->start, transfer.rows,
                             dram->rowBytes, dram->rowBytes, dram->stride};
  return DMALoad2DCommand{transfer.tensor.id, transfer.buffer.requirement.id,
                          dram->start, transfer.buffer.address, transfer.rows,
                          dram->rowBytes, dram->stride, dram->rowBytes};
}

llvm::Error verifyLive(const BufferAllocation &buffer, int64_t event) {
  const auto &req = buffer.requirement;
  if (event < req.liveBegin || event >= req.liveEnd)
    return llvm::createStringError("SRAM buffer is not live at command event");
  return llvm::Error::success();
}

llvm::Error verifyTransfer(const TileEvent &event, const MatMulTensors &tensors,
                           int64_t tensorId, int64_t bufferId, int64_t dramStart,
                           int64_t sramStart, int64_t rows, int64_t rowBytes,
                           int64_t dramStride, int64_t sramStride) {
  auto transfer = getTransfer(event, tensors);
  if (tensorId != transfer.tensor.id)
    return llvm::createStringError("DMA tensor_id does not match operand or result");
  if (bufferId != transfer.buffer.requirement.id)
    return llvm::createStringError("DMA buffer_id does not match tile binding");
  if (auto error = verifyLive(transfer.buffer, event.index))
    return error;
  if (auto error = verifyDMA(transfer.tensor, transfer.buffer, dramStart, sramStart,
                             rows, rowBytes, dramStride, sramStride))
    return error;
  auto expected = getDRAMWindow(transfer.tensor, transfer.row, transfer.col,
                                transfer.rows, transfer.cols);
  if (!expected)
    return expected.takeError();
  if (dramStart != expected->start || sramStart != transfer.buffer.address ||
      rows != transfer.rows || rowBytes != expected->rowBytes ||
      dramStride != expected->stride || sramStride != expected->rowBytes)
    return llvm::createStringError("DMA geometry does not match tile region");
  return llvm::Error::success();
}

llvm::Error verifyPayload(const TileEvent &event, const MatMulTensors &tensors,
                          const CommandPayload &payload) {
  const TileDesc &tile = event.tile;
  switch (event.kind) {
  case EventKind::ZeroC:
    if (const auto *command = std::get_if<ZeroCommand>(&payload)) {
      if (command->bufferId != event.c.requirement.id ||
          command->dstSram != event.c.address || command->rows != tile.mSize ||
          command->cols != tile.nSize)
        return llvm::createStringError("ZERO does not match tile C binding");
      if (auto error = verifyLive(event.c, event.index))
        return error;
      return verifyCompactSRAM(event.c, command->dstSram, command->rows,
                               command->cols);
    }
    break;
  case EventKind::LoadA:
  case EventKind::LoadB:
    if (const auto *command = std::get_if<DMALoad2DCommand>(&payload))
      return verifyTransfer(event, tensors, command->tensorId,
                             command->bufferId, command->srcDram,
                             command->dstSram, command->rows, command->rowBytes,
                             command->srcStrideBytes, command->dstStrideBytes);
    break;
  case EventKind::Compute:
    if (const auto *command = std::get_if<MatMulTileCommand>(&payload)) {
      if (command->aBufferId != event.a.requirement.id ||
          command->bBufferId != event.b.requirement.id ||
          command->cBufferId != event.c.requirement.id ||
          command->aSram != event.a.address ||
          command->bSram != event.b.address ||
          command->cSram != event.c.address || command->m != tile.mSize ||
          command->n != tile.nSize || command->k != tile.kSize ||
          !command->accumulate)
        return llvm::createStringError(
            "MATMUL_TILE does not match tile dimensions or bindings");
      for (const BufferAllocation *buffer : {&event.a, &event.b, &event.c})
        if (auto error = verifyLive(*buffer, event.index))
          return error;
      if (auto error = verifyCompactSRAM(event.a, command->aSram, command->m,
                                         command->k))
        return error;
      if (auto error = verifyCompactSRAM(event.b, command->bSram, command->k,
                                         command->n))
        return error;
      return verifyCompactSRAM(event.c, command->cSram, command->m, command->n);
    }
    break;
  case EventKind::StoreC:
    if (const auto *command = std::get_if<DMAStore2DCommand>(&payload))
      return verifyTransfer(event, tensors, command->tensorId,
                             command->bufferId, command->dstDram,
                             command->srcSram, command->rows, command->rowBytes,
                             command->dstStrideBytes, command->srcStrideBytes);
    break;
  }
  return llvm::createStringError("opcode does not match tile event");
}

} // namespace

llvm::Expected<std::vector<CommandRecord>>
mlir::npu::generateCommands(llvm::ArrayRef<TileDesc> tiles, const MemoryPlan &memory,
                            const MatMulTensors &tensors) {
  if (tiles.empty() || tiles.size() > maxTileRecords)
    return llvm::createStringError("invalid tile count for command plan");
  std::vector<CommandRecord> commands;
  commands.reserve(commandCount(tiles));
  if (auto error = visitEvents(tiles, memory, [&](const TileEvent &event) -> llvm::Error {
        auto payload = buildPayload(event, tensors);
        if (!payload)
          return payload.takeError();
        commands.push_back({event.index, event.tileIndex, std::move(*payload)});
        return llvm::Error::success();
      }))
    return error;
  return commands;
}

llvm::Error mlir::npu::verifyCommands(llvm::ArrayRef<TileDesc> tiles,
                                      const MemoryPlan &memory,
                                      const MatMulTensors &tensors,
                                      llvm::ArrayRef<CommandRecord> commands) {
  if (tiles.empty() || tiles.size() > maxTileRecords)
    return llvm::createStringError("invalid tile count for command plan");
  if (commands.size() != commandCount(tiles))
    return llvm::createStringError("command count does not match tile events");
  return visitEvents(tiles, memory, [&](const TileEvent &event) -> llvm::Error {
    const CommandRecord &command = commands[event.index];
    if (command.eventIndex != event.index || command.tileIndex != event.tileIndex)
      return llvm::createStringError(
          "event_index or tile_index does not match tile event");
    return verifyPayload(event, tensors, command.payload);
  });
}
