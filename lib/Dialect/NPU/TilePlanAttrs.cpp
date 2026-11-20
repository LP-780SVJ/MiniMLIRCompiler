#include "NPU/TilePlanAttrs.h"

#include "mlir/IR/Builders.h"

#include <iterator>
#include <utility>

using namespace mlir;
using namespace mlir::npu;

namespace {

constexpr std::pair<const char *, int64_t TileDesc::*> integerFields[] = {
    {"m_offset", &TileDesc::mOffset}, {"n_offset", &TileDesc::nOffset},
    {"k_offset", &TileDesc::kOffset}, {"m_size", &TileDesc::mSize},
    {"n_size", &TileDesc::nSize},     {"k_size", &TileDesc::kSize},
    {"output_tile_id", &TileDesc::outputTileId},
};
constexpr std::pair<const char *, bool TileDesc::*> booleanFields[] = {
    {"is_first_k", &TileDesc::isFirstK},
    {"is_last_k", &TileDesc::isLastK},
};

}

DictionaryAttr mlir::npu::encodeTileDesc(Builder &builder,
                                        const TileDesc &tile) {
  SmallVector<NamedAttribute, 9> fields;
  for (auto [name, member] : integerFields)
    fields.push_back(
        builder.getNamedAttr(name, builder.getI64IntegerAttr(tile.*member)));
  for (auto [name, member] : booleanFields)
    fields.push_back(builder.getNamedAttr(name, builder.getBoolAttr(tile.*member)));
  return builder.getDictionaryAttr(fields);
}

llvm::Expected<TileDesc> mlir::npu::decodeTileDesc(DictionaryAttr attr) {
  TileDesc tile{};
  for (auto [name, member] : integerFields) {
    auto value = dyn_cast_or_null<IntegerAttr>(attr.get(name));
    if (!value || !value.getType().isSignlessInteger(64))
      return llvm::createStringError("field '%s' must be i64", name);
    tile.*member = value.getInt();
  }
  for (auto [name, member] : booleanFields) {
    auto value = dyn_cast_or_null<BoolAttr>(attr.get(name));
    if (!value)
      return llvm::createStringError("field '%s' must be bool", name);
    tile.*member = value.getValue();
  }
  if (attr.size() != std::size(integerFields) + std::size(booleanFields))
    return llvm::createStringError("unexpected tile descriptor fields");
  return tile;
}
