#include "NPU/CommandPlanAttrs.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

using namespace mlir;
using namespace mlir::npu;

namespace {

template <typename Record>
using IntegerField = std::pair<const char *, int64_t Record::*>;

constexpr IntegerField<DRAMLayout> layoutFields[] = {
    {"tensor_alignment_bytes", &DRAMLayout::alignmentBytes},
    {"total_bytes", &DRAMLayout::totalBytes},
};

constexpr IntegerField<DRAMTensorDesc> tensorFields[] = {
    {"id", &DRAMTensorDesc::id},
    {"source_index", &DRAMTensorDesc::sourceIndex},
    {"base", &DRAMTensorDesc::base},
    {"rows", &DRAMTensorDesc::rows},
    {"cols", &DRAMTensorDesc::cols},
    {"row_stride_bytes", &DRAMTensorDesc::rowStrideBytes},
    {"size_bytes", &DRAMTensorDesc::sizeBytes},
};

constexpr IntegerField<CommandRecord> commandFields[] = {
    {"event_index", &CommandRecord::eventIndex},
    {"tile_index", &CommandRecord::tileIndex},
};

template <typename Payload> struct Schema;

template <> struct Schema<ZeroCommand> {
  static constexpr char opcode[] = "ZERO";
  static constexpr IntegerField<ZeroCommand> fields[] = {
      {"buffer_id", &ZeroCommand::bufferId},
      {"dst_sram", &ZeroCommand::dstSram},
      {"rows", &ZeroCommand::rows},
      {"cols", &ZeroCommand::cols},
  };
};

template <> struct Schema<DMALoad2DCommand> {
  static constexpr char opcode[] = "DMA_LOAD_2D";
  static constexpr IntegerField<DMALoad2DCommand> fields[] = {
      {"tensor_id", &DMALoad2DCommand::tensorId},
      {"buffer_id", &DMALoad2DCommand::bufferId},
      {"src_dram", &DMALoad2DCommand::srcDram},
      {"dst_sram", &DMALoad2DCommand::dstSram},
      {"rows", &DMALoad2DCommand::rows},
      {"row_bytes", &DMALoad2DCommand::rowBytes},
      {"src_stride_bytes", &DMALoad2DCommand::srcStrideBytes},
      {"dst_stride_bytes", &DMALoad2DCommand::dstStrideBytes},
  };
};

template <> struct Schema<MatMulTileCommand> {
  static constexpr char opcode[] = "MATMUL_TILE";
  static constexpr IntegerField<MatMulTileCommand> fields[] = {
      {"a_buffer_id", &MatMulTileCommand::aBufferId},
      {"b_buffer_id", &MatMulTileCommand::bBufferId},
      {"c_buffer_id", &MatMulTileCommand::cBufferId},
      {"a_sram", &MatMulTileCommand::aSram},
      {"b_sram", &MatMulTileCommand::bSram},
      {"c_sram", &MatMulTileCommand::cSram},
      {"m", &MatMulTileCommand::m},
      {"n", &MatMulTileCommand::n},
      {"k", &MatMulTileCommand::k},
  };
};

template <> struct Schema<DMAStore2DCommand> {
  static constexpr char opcode[] = "DMA_STORE_2D";
  static constexpr IntegerField<DMAStore2DCommand> fields[] = {
      {"tensor_id", &DMAStore2DCommand::tensorId},
      {"buffer_id", &DMAStore2DCommand::bufferId},
      {"src_sram", &DMAStore2DCommand::srcSram},
      {"dst_dram", &DMAStore2DCommand::dstDram},
      {"rows", &DMAStore2DCommand::rows},
      {"row_bytes", &DMAStore2DCommand::rowBytes},
      {"src_stride_bytes", &DMAStore2DCommand::srcStrideBytes},
      {"dst_stride_bytes", &DMAStore2DCommand::dstStrideBytes},
  };
};

template <typename Record, size_t Count>
void encodeIntegers(Builder &builder, SmallVectorImpl<NamedAttribute> &attrs,
                    const Record &record,
                    const IntegerField<Record> (&fields)[Count]) {
  for (auto [name, member] : fields)
    attrs.push_back(
        builder.getNamedAttr(name, builder.getI64IntegerAttr(record.*member)));
}

llvm::Expected<int64_t> decodeNonnegativeI64(Attribute attr, const char *name) {
  auto value = dyn_cast_or_null<IntegerAttr>(attr);
  if (!value || !value.getType().isSignlessInteger(64) || value.getInt() < 0)
    return llvm::createStringError("field '%s' must be a nonnegative i64", name);
  return value.getInt();
}

template <typename Record, size_t Count>
llvm::Error decodeIntegers(DictionaryAttr attr, Record &record,
                           const IntegerField<Record> (&fields)[Count]) {
  for (auto [name, member] : fields) {
    auto value = decodeNonnegativeI64(attr.get(name), name);
    if (!value)
      return value.takeError();
    record.*member = *value;
  }
  return llvm::Error::success();
}

StringRef encodeStorageKind(TensorStorageKind kind) {
  switch (kind) {
  case TensorStorageKind::Input:
    return "input";
  case TensorStorageKind::Constant:
    return "constant";
  case TensorStorageKind::Intermediate:
    return "intermediate";
  case TensorStorageKind::Output:
    return "output";
  }
  llvm_unreachable("unknown TensorStorageKind");
}

DictionaryAttr encodeTensor(Builder &builder, const DRAMTensorDesc &tensor) {
  SmallVector<NamedAttribute, 10> fields;
  encodeIntegers(builder, fields, tensor, tensorFields);
  fields.push_back(builder.getNamedAttr(
      "kind", builder.getStringAttr(encodeStorageKind(tensor.kind))));
  fields.push_back(builder.getNamedAttr("dtype", builder.getStringAttr("float32")));
  fields.push_back(
      builder.getNamedAttr("layout", builder.getStringAttr("row_major")));
  return builder.getDictionaryAttr(fields);
}

ArrayAttr encodeIds(Builder &builder, llvm::ArrayRef<int64_t> ids) {
  SmallVector<Attribute> attrs;
  attrs.reserve(ids.size());
  for (int64_t id : ids)
    attrs.push_back(builder.getI64IntegerAttr(id));
  return builder.getArrayAttr(attrs);
}

llvm::Error decodeIds(DictionaryAttr attr, const char *name,
                      std::vector<int64_t> &ids) {
  auto array = dyn_cast_or_null<ArrayAttr>(attr.get(name));
  if (!array)
    return llvm::createStringError("field '%s' must be an array", name);
  if (array.size() > maxDRAMRecords)
    return llvm::createStringError("field '%s' exceeds DRAM record limit", name);
  ids.reserve(array.size());
  for (Attribute entry : array) {
    auto id = decodeNonnegativeI64(entry, name);
    if (!id)
      return id.takeError();
    ids.push_back(*id);
  }
  return llvm::Error::success();
}

DictionaryAttr encodeCommand(Builder &builder, const CommandRecord &record) {
  SmallVector<NamedAttribute, 13> fields;
  encodeIntegers(builder, fields, record, commandFields);
  std::visit(
      [&](const auto &payload) {
        using Payload = std::decay_t<decltype(payload)>;
        fields.push_back(builder.getNamedAttr(
            "opcode", builder.getStringAttr(Schema<Payload>::opcode)));
        encodeIntegers(builder, fields, payload, Schema<Payload>::fields);
        if constexpr (std::is_same_v<Payload, MatMulTileCommand>)
          fields.push_back(builder.getNamedAttr(
              "accumulate", builder.getBoolAttr(payload.accumulate)));
      },
      record.payload);
  return builder.getDictionaryAttr(fields);
}

template <typename Payload>
llvm::Expected<CommandRecord> decodeCommandAs(DictionaryAttr attr) {
  CommandRecord record{};
  Payload payload{};
  if (auto error = decodeIntegers(attr, record, commandFields))
    return error;
  if (auto error = decodeIntegers(attr, payload, Schema<Payload>::fields))
    return error;
  size_t fieldCount =
      std::size(commandFields) + std::size(Schema<Payload>::fields) + 1;
  if constexpr (std::is_same_v<Payload, MatMulTileCommand>) {
    auto accumulate = dyn_cast_or_null<BoolAttr>(attr.get("accumulate"));
    if (!accumulate)
      return llvm::createStringError("field 'accumulate' must be bool");
    payload.accumulate = accumulate.getValue();
    ++fieldCount;
  }
  if (attr.size() != fieldCount)
    return llvm::createStringError("unexpected fields for %s",
                                   Schema<Payload>::opcode);
  record.payload = payload;
  return record;
}

llvm::Expected<CommandRecord> decodeCommand(DictionaryAttr attr) {
  auto opcode = dyn_cast_or_null<StringAttr>(attr.get("opcode"));
  if (!opcode)
    return llvm::createStringError("field 'opcode' must be a string");
  StringRef name = opcode.getValue();
  if (name == Schema<ZeroCommand>::opcode)
    return decodeCommandAs<ZeroCommand>(attr);
  if (name == Schema<DMALoad2DCommand>::opcode)
    return decodeCommandAs<DMALoad2DCommand>(attr);
  if (name == Schema<MatMulTileCommand>::opcode)
    return decodeCommandAs<MatMulTileCommand>(attr);
  if (name == Schema<DMAStore2DCommand>::opcode)
    return decodeCommandAs<DMAStore2DCommand>(attr);
  return llvm::createStringError(llvm::Twine("unsupported command opcode '") +
                                 name + "'");
}

}

DictionaryAttr mlir::npu::encodeDRAMLayout(Builder &builder,
                                         const DRAMLayout &layout) {
  SmallVector<NamedAttribute, 7> fields;
  encodeIntegers(builder, fields, layout, layoutFields);
  fields.push_back(
      builder.getNamedAttr("schema_version", builder.getI64IntegerAttr(1)));
  SmallVector<Attribute> tensors;
  tensors.reserve(layout.tensors.size());
  for (const auto &tensor : layout.tensors)
    tensors.push_back(encodeTensor(builder, tensor));
  fields.push_back(builder.getNamedAttr("tensors", builder.getArrayAttr(tensors)));
  fields.push_back(
      builder.getNamedAttr("input_ids", encodeIds(builder, layout.inputIds)));
  fields.push_back(builder.getNamedAttr("constant_ids",
                                        encodeIds(builder, layout.constantIds)));
  fields.push_back(
      builder.getNamedAttr("output_ids", encodeIds(builder, layout.outputIds)));
  return builder.getDictionaryAttr(fields);
}

llvm::Expected<DRAMTensorDesc>
mlir::npu::decodeDRAMTensorDesc(DictionaryAttr attr) {
  if (!attr)
    return llvm::createStringError("DRAM tensor descriptor must be a dictionary");
  DRAMTensorDesc tensor{};
  if (auto error = decodeIntegers(attr, tensor, tensorFields))
    return error;
  auto dtype = dyn_cast_or_null<StringAttr>(attr.get("dtype"));
  if (!dtype || dtype.getValue() != "float32")
    return llvm::createStringError("field 'dtype' must be 'float32'");
  auto layout = dyn_cast_or_null<StringAttr>(attr.get("layout"));
  if (!layout || layout.getValue() != "row_major")
    return llvm::createStringError("field 'layout' must be 'row_major'");

  auto kind = dyn_cast_or_null<StringAttr>(attr.get("kind"));
  if (!kind)
    return llvm::createStringError("field 'kind' must be a string");
  if (kind.getValue() == "input")
    tensor.kind = TensorStorageKind::Input;
  else if (kind.getValue() == "constant")
    tensor.kind = TensorStorageKind::Constant;
  else if (kind.getValue() == "intermediate")
    tensor.kind = TensorStorageKind::Intermediate;
  else if (kind.getValue() == "output")
    tensor.kind = TensorStorageKind::Output;
  else
    return llvm::createStringError("invalid DRAM tensor storage kind");
  if (attr.size() != std::size(tensorFields) + 3)
    return llvm::createStringError("unexpected DRAM tensor descriptor fields");
  return tensor;
}

llvm::Expected<DRAMLayout> mlir::npu::decodeDRAMLayout(DictionaryAttr attr) {
  if (!attr)
    return llvm::createStringError("npu.dram_layout must be a dictionary");
  auto version = decodeNonnegativeI64(attr.get("schema_version"), "schema_version");
  if (!version)
    return version.takeError();
  if (*version != 1)
    return llvm::createStringError("unsupported DRAM layout schema_version");

  DRAMLayout layout{};
  if (auto error = decodeIntegers(attr, layout, layoutFields))
    return error;
  auto tensors = dyn_cast_or_null<ArrayAttr>(attr.get("tensors"));
  if (!tensors)
    return llvm::createStringError("field 'tensors' must be an array");
  if (tensors.size() > maxDRAMRecords)
    return llvm::createStringError("field 'tensors' exceeds DRAM record limit");
  layout.tensors.reserve(tensors.size());
  for (auto [index, entry] : llvm::enumerate(tensors)) {
    auto tensor = decodeDRAMTensorDesc(dyn_cast<DictionaryAttr>(entry));
    if (!tensor)
      return llvm::createStringError("DRAM tensor #%zu: %s", index,
                                     llvm::toString(tensor.takeError()).c_str());
    layout.tensors.push_back(*tensor);
  }
  if (auto error = decodeIds(attr, "input_ids", layout.inputIds))
    return error;
  if (auto error = decodeIds(attr, "constant_ids", layout.constantIds))
    return error;
  if (auto error = decodeIds(attr, "output_ids", layout.outputIds))
    return error;
  if (attr.size() != std::size(layoutFields) + 5)
    return llvm::createStringError("unexpected DRAM layout fields");
  return layout;
}

ArrayAttr mlir::npu::encodeCommands(Builder &builder,
                                    llvm::ArrayRef<CommandRecord> commands) {
  SmallVector<Attribute> result;
  result.reserve(commands.size());
  for (const auto &command : commands)
    result.push_back(encodeCommand(builder, command));
  return builder.getArrayAttr(result);
}

llvm::Expected<std::vector<CommandRecord>>
mlir::npu::decodeCommands(ArrayAttr attr) {
  if (!attr)
    return llvm::createStringError("commands must be an array");
  if (attr.size() > maxCommandRecords)
    return llvm::createStringError("commands exceed command record limit");
  std::vector<CommandRecord> result;
  result.reserve(attr.size());
  for (auto [index, entry] : llvm::enumerate(attr)) {
    auto dict = dyn_cast<DictionaryAttr>(entry);
    if (!dict)
      return llvm::createStringError("command #%zu must be a dictionary", index);
    auto command = decodeCommand(dict);
    if (!command)
      return llvm::createStringError("command #%zu: %s", index,
                                     llvm::toString(command.takeError()).c_str());
    result.push_back(*command);
  }
  return result;
}
