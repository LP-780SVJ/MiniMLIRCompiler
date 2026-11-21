#include "NPU/MemoryPlanAttrs.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::npu;

namespace {

template <typename Record>
using IntegerField = std::pair<const char *, int64_t Record::*>;

constexpr IntegerField<SRAMConfig> configFields[] = {
    {"sram_capacity_bytes", &SRAMConfig::capacityBytes},
    {"reserved_bytes", &SRAMConfig::reservedBytes},
    {"alignment_bytes", &SRAMConfig::alignmentBytes},
};

constexpr IntegerField<MemoryPlan> statisticFields[] = {
    {"peak_live_bytes", &MemoryPlan::peakLiveBytes},
    {"arena_high_water_bytes", &MemoryPlan::arenaHighWaterBytes},
};

constexpr IntegerField<BufferRequirement> requirementFields[] = {
    {"id", &BufferRequirement::id},
    {"tile_index", &BufferRequirement::tileIndex},
    {"output_tile_id", &BufferRequirement::outputTileId},
    {"raw_bytes", &BufferRequirement::rawBytes},
    {"allocated_bytes", &BufferRequirement::allocatedBytes},
    {"alignment_bytes", &BufferRequirement::alignmentBytes},
    {"live_begin", &BufferRequirement::liveBegin},
    {"live_end", &BufferRequirement::liveEnd},
};

constexpr IntegerField<BufferAllocation> allocationFields[] = {
    {"address", &BufferAllocation::address},
};

constexpr IntegerField<TileBufferBinding> bindingFields[] = {
    {"tile_index", &TileBufferBinding::tileIndex},
    {"a_buffer_id", &TileBufferBinding::aBufferId},
    {"b_buffer_id", &TileBufferBinding::bBufferId},
    {"c_buffer_id", &TileBufferBinding::cBufferId},
};

template <typename Record, size_t Count>
void encodeIntegers(Builder &builder, SmallVectorImpl<NamedAttribute> &attrs,
                    const Record &record,
                    const IntegerField<Record> (&fields)[Count]) {
  for (auto [name, member] : fields)
    attrs.push_back(
        builder.getNamedAttr(name, builder.getI64IntegerAttr(record.*member)));
}

template <typename Record, size_t Count>
llvm::Error decodeIntegers(DictionaryAttr attr, Record &record,
                           const IntegerField<Record> (&fields)[Count]) {
  for (auto [name, member] : fields) {
    auto value = dyn_cast_or_null<IntegerAttr>(attr.get(name));
    if (!value || !value.getType().isSignlessInteger(64) || value.getInt() < 0)
      return llvm::createStringError("field '%s' must be a nonnegative i64",
                                     name);
    record.*member = value.getInt();
  }
  return llvm::Error::success();
}

StringRef encodeRole(BufferRole role) {
  switch (role) {
  case BufferRole::A:
    return "A";
  case BufferRole::B:
    return "B";
  case BufferRole::C:
    return "C";
  }
  llvm_unreachable("unknown BufferRole");
}

DictionaryAttr encodeAllocation(Builder &builder,
                                const BufferAllocation &allocation) {
  SmallVector<NamedAttribute, 11> fields;
  encodeIntegers(builder, fields, allocation.requirement, requirementFields);
  encodeIntegers(builder, fields, allocation, allocationFields);
  fields.push_back(builder.getNamedAttr(
      "role", builder.getStringAttr(encodeRole(allocation.requirement.role))));
  fields.push_back(builder.getNamedAttr(
      "reused_address", builder.getBoolAttr(allocation.reusedAddress)));
  return builder.getDictionaryAttr(fields);
}

llvm::Expected<BufferAllocation> decodeAllocation(DictionaryAttr attr) {
  BufferAllocation record{};
  if (auto error = decodeIntegers(attr, record.requirement, requirementFields))
    return error;
  if (auto error = decodeIntegers(attr, record, allocationFields))
    return error;
  auto role = dyn_cast_or_null<StringAttr>(attr.get("role"));
  if (role && role.getValue() == "A")
    record.requirement.role = BufferRole::A;
  else if (role && role.getValue() == "B")
    record.requirement.role = BufferRole::B;
  else if (role && role.getValue() == "C")
    record.requirement.role = BufferRole::C;
  else
    return llvm::createStringError("field 'role' must be \"A\", \"B\", or \"C\"");
  auto reused = dyn_cast_or_null<BoolAttr>(attr.get("reused_address"));
  if (!reused)
    return llvm::createStringError("field 'reused_address' must be bool");
  record.reusedAddress = reused.getValue();
  if (attr.size() != std::size(requirementFields) +
                         std::size(allocationFields) + 2)
    return llvm::createStringError("unexpected allocation fields");
  return record;
}

DictionaryAttr encodeBinding(Builder &builder,
                             const TileBufferBinding &binding) {
  SmallVector<NamedAttribute, 4> fields;
  encodeIntegers(builder, fields, binding, bindingFields);
  return builder.getDictionaryAttr(fields);
}

llvm::Expected<TileBufferBinding> decodeBinding(DictionaryAttr attr) {
  TileBufferBinding record{};
  if (auto error = decodeIntegers(attr, record, bindingFields))
    return error;
  if (attr.size() != std::size(bindingFields))
    return llvm::createStringError("unexpected tile binding fields");
  return record;
}

template <typename Record>
ArrayAttr encodeRecords(Builder &builder, const std::vector<Record> &records,
                        DictionaryAttr (*encode)(Builder &, const Record &)) {
  SmallVector<Attribute> attrs;
  attrs.reserve(records.size());
  for (const auto &record : records)
    attrs.push_back(encode(builder, record));
  return builder.getArrayAttr(attrs);
}

template <typename Record>
llvm::Error decodeRecords(DictionaryAttr attr, const char *name,
                          size_t maxRecords, std::vector<Record> &records,
                          llvm::Expected<Record> (*decode)(DictionaryAttr)) {
  auto array = dyn_cast_or_null<ArrayAttr>(attr.get(name));
  if (!array)
    return llvm::createStringError("field '%s' must be an array", name);
  if (array.size() > maxRecords)
    return llvm::createStringError("field '%s' exceeds record limit", name);
  records.reserve(array.size());
  for (size_t index = 0; index < array.size(); ++index) {
    auto dict = dyn_cast<DictionaryAttr>(array[index]);
    if (!dict)
      return llvm::createStringError("%s #%zu must be a dictionary", name,
                                     index);
    auto record = decode(dict);
    if (!record)
      return llvm::createStringError(
          "%s #%zu: %s", name, index,
          llvm::toString(record.takeError()).c_str());
    records.push_back(*record);
  }
  return llvm::Error::success();
}

}

DictionaryAttr mlir::npu::encodeMemoryPlan(Builder &builder,
                                          const MemoryPlan &plan) {
  SmallVector<NamedAttribute, 8> fields;
  encodeIntegers(builder, fields, plan.config, configFields);
  encodeIntegers(builder, fields, plan, statisticFields);
  fields.push_back(builder.getNamedAttr(
      "schedule_contract", builder.getStringAttr(serialSRAMSchedule)));
  fields.push_back(builder.getNamedAttr(
      "allocations", encodeRecords(builder, plan.allocations, encodeAllocation)));
  fields.push_back(builder.getNamedAttr(
      "tile_bindings", encodeRecords(builder, plan.bindings, encodeBinding)));
  return builder.getDictionaryAttr(fields);
}

llvm::Expected<MemoryPlan> mlir::npu::decodeMemoryPlan(DictionaryAttr attr) {
  if (!attr)
    return llvm::createStringError("missing memory_plan dictionary");
  MemoryPlan plan{};
  if (auto error = decodeIntegers(attr, plan.config, configFields))
    return error;
  if (auto error = decodeIntegers(attr, plan, statisticFields))
    return error;
  auto schedule = dyn_cast_or_null<StringAttr>(attr.get("schedule_contract"));
  if (!schedule || schedule.getValue() != serialSRAMSchedule)
    return llvm::createStringError("field 'schedule_contract' must be '%s'",
                                   serialSRAMSchedule);
  if (attr.size() != std::size(configFields) + std::size(statisticFields) + 3)
    return llvm::createStringError("unexpected memory plan fields");
  if (auto error = decodeRecords(attr, "allocations", 3 * maxTileRecords,
                                 plan.allocations, decodeAllocation))
    return error;
  if (auto error = decodeRecords(attr, "tile_bindings", maxTileRecords,
                                 plan.bindings, decodeBinding))
    return error;
  return plan;
}
