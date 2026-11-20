#ifndef NPU_TILEPLANATTRS_H
#define NPU_TILEPLANATTRS_H

#include "NPU/TilePlan.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class Builder;

namespace npu {

inline constexpr char virtualNPUTarget[] = "virtual-npu-v1";

DictionaryAttr encodeTileDesc(Builder &builder, const TileDesc &tile);
llvm::Expected<TileDesc> decodeTileDesc(DictionaryAttr attr);

}
}

#endif
