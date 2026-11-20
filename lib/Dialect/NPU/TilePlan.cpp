#include "NPU/TilePlan.h"

#include <algorithm>
#include <cassert>

using namespace mlir::npu;

llvm::Expected<TilePlan> TilePlan::create(TileShape shape, TileShape limits) {
  if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0)
    return llvm::createStringError("MatMul dimensions must be positive");
  if (limits.m <= 0 || limits.n <= 0 || limits.k <= 0)
    return llvm::createStringError("native tile limits must be positive");

  int64_t mTiles = 1 + (shape.m - 1) / limits.m;
  int64_t nTiles = 1 + (shape.n - 1) / limits.n;
  int64_t kTiles = 1 + (shape.k - 1) / limits.k;
  if (mTiles > maxTileRecords / nTiles ||
      mTiles * nTiles > maxTileRecords / kTiles)
    return llvm::createStringError("tile count exceeds limit of %lld",
                                   static_cast<long long>(maxTileRecords));
  return TilePlan(shape, limits, nTiles, kTiles, mTiles * nTiles * kTiles);
}

TileDesc TilePlan::getTile(int64_t index) const {
  assert(index >= 0 && index < count);
  int64_t outputId = index / kTiles;
  int64_t kIndex = index % kTiles;
  int64_t mOffset = (outputId / nTiles) * limits.m;
  int64_t nOffset = (outputId % nTiles) * limits.n;
  int64_t kOffset = kIndex * limits.k;
  return {mOffset, nOffset, kOffset,
          std::min(limits.m, shape.m - mOffset),
          std::min(limits.n, shape.n - nOffset),
          std::min(limits.k, shape.k - kOffset),
          outputId, kIndex == 0, kIndex == kTiles - 1};
}
