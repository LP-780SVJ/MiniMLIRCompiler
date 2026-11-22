#include "NPU/CommandEffects.h"

#include "llvm/Support/MathExtras.h"

using namespace mlir::npu;

namespace {

llvm::Error add2DAccess(llvm::SmallVectorImpl<MemoryAccess> &accesses,
                        MemorySpace space, int64_t base, int64_t rows,
                        int64_t rowBytes, int64_t strideBytes) {
  auto end = checkedDMAEnd(base, rows, rowBytes, strideBytes);
  if (!end)
    return end.takeError();
  accesses.push_back({space, {base, *end}});
  return llvm::Error::success();
}

llvm::Error addSRAMTile(llvm::SmallVectorImpl<MemoryAccess> &accesses,
                       int64_t base, int64_t rows, int64_t cols) {
  int64_t rowBytes;
  if (llvm::MulOverflow(cols, int64_t{4}, rowBytes))
    return llvm::createStringError("SRAM tile byte range overflow");
  return add2DAccess(accesses, MemorySpace::SRAM, base, rows, rowBytes, rowBytes);
}

}

llvm::Expected<CommandEffects>
mlir::npu::getCommandEffects(const CommandRecord &command) {
  CommandEffects effects;
  const auto &payload = command.payload;
  if (const auto *zero = std::get_if<ZeroCommand>(&payload)) {
    if (auto error =
            addSRAMTile(effects.writes, zero->dstSram, zero->rows, zero->cols))
      return error;
  } else if (const auto *load = std::get_if<DMALoad2DCommand>(&payload)) {
    if (auto error = add2DAccess(effects.reads, MemorySpace::DRAM, load->srcDram,
                                 load->rows, load->rowBytes, load->srcStrideBytes))
      return error;
    if (auto error = add2DAccess(effects.writes, MemorySpace::SRAM, load->dstSram,
                                 load->rows, load->rowBytes, load->dstStrideBytes))
      return error;
  } else if (const auto *matmul = std::get_if<MatMulTileCommand>(&payload)) {
    if (!matmul->accumulate)
      return llvm::createStringError("MATMUL_TILE must accumulate");
    if (auto error = addSRAMTile(effects.reads, matmul->aSram, matmul->m, matmul->k))
      return error;
    if (auto error = addSRAMTile(effects.reads, matmul->bSram, matmul->k, matmul->n))
      return error;
    if (auto error = addSRAMTile(effects.reads, matmul->cSram, matmul->m, matmul->n))
      return error;
    effects.writes.push_back(effects.reads.back());
  } else if (const auto *store = std::get_if<DMAStore2DCommand>(&payload)) {
    if (auto error = add2DAccess(effects.reads, MemorySpace::SRAM, store->srcSram,
                                 store->rows, store->rowBytes, store->srcStrideBytes))
      return error;
    if (auto error = add2DAccess(effects.writes, MemorySpace::DRAM, store->dstDram,
                                 store->rows, store->rowBytes, store->dstStrideBytes))
      return error;
  }
  return effects;
}
