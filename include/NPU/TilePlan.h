#ifndef NPU_TILEPLAN_H
#define NPU_TILEPLAN_H

#include "llvm/Support/Error.h"
#include <cstdint>

namespace mlir::npu {

struct TileShape {
  int64_t m, n, k;
};

struct TileDesc {
  int64_t mOffset, nOffset, kOffset;
  int64_t mSize, nSize, kSize;
  int64_t outputTileId;
  bool isFirstK, isLastK;
};

inline constexpr int64_t maxTileRecords = 65536;

class TilePlan {
public:
  static llvm::Expected<TilePlan> create(TileShape shape, TileShape limits);

  int64_t size() const { return count; }
  TileDesc getTile(int64_t index) const;

private:
  TilePlan(TileShape shape, TileShape limits, int64_t nTiles, int64_t kTiles,
           int64_t count)
      : shape(shape), limits(limits), nTiles(nTiles), kTiles(kTiles),
        count(count) {}

  TileShape shape;
  TileShape limits;
  int64_t nTiles, kTiles, count;
};

}

#endif
