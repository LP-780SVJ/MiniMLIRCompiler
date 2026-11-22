#include "NPU/SchedulePlanAttrs.h"

#include "NPU/CommandPlanAttrs.h"
#include "NPU/MemoryPlanAttrs.h"
#include "NPU/NPUOps.h"
#include "NPU/TilePlanAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

#include <utility>

using namespace mlir;
using namespace mlir::npu;

namespace {

llvm::Expected<int64_t> readIndex(Attribute attr, StringRef field) {
  auto value = dyn_cast_or_null<IntegerAttr>(attr);
  if (!value || !value.getType().isSignlessInteger(64) || value.getInt() < 0)
    return llvm::createStringError(llvm::Twine(field) + " must be a nonnegative i64");
  return value.getInt();
}

}

llvm::Expected<ScheduleInput> mlir::npu::readScheduleInput(ModuleOp module) {
  auto layout = decodeDRAMLayout(module->getAttrOfType<DictionaryAttr>(dramLayoutAttr));
  if (!layout)
    return layout.takeError();
  ScheduleInput input{std::move(*layout), {}};
  llvm::DenseMap<Value, int64_t> ids;
  auto expected = buildDRAMLayout(module, DRAMLayoutStage::Commands, ids);
  if (!expected)
    return expected.takeError();
  Builder builder(module.getContext());
  if (encodeDRAMLayout(builder, *expected) != module->getAttr(dramLayoutAttr))
    return llvm::createStringError("schedule DRAM layout does not match SSA");

  int64_t total = 0;
  for (Operation &op : module.getBody()->front().getRegion(0).front()) {
    auto matmul = dyn_cast<CommandMatMulOp>(op);
    if (!matmul)
      continue;
    if (failed(mlir::verify(&op, false)))
      return llvm::createStringError("schedule input contains invalid command MatMul");
    ScheduleMatMul group{};
    group.reductionSize = matmul.getLhs().getType().getDimSize(1);
    Value values[] = {matmul.getLhs(), matmul.getRhs(), matmul.getResult()};
    int64_t actual[] = {matmul.getLhsTensorIdAttr().getInt(),
                         matmul.getRhsTensorIdAttr().getInt(),
                         matmul.getResultTensorIdAttr().getInt()};
    for (size_t index = 0; index < group.tensorIds.size(); ++index) {
      auto id = ids.find(values[index]);
      if (id == ids.end() || id->second != actual[index])
        return llvm::createStringError("schedule tensor ID does not match SSA binding");
      group.tensorIds[index] = id->second;
    }
    for (Attribute attr : matmul.getTiles()) {
      auto tile = decodeTileDesc(cast<DictionaryAttr>(attr));
      if (!tile)
        return tile.takeError();
      group.tiles.push_back(*tile);
    }
    auto memory = decodeMemoryPlan(matmul.getMemoryPlan());
    if (!memory)
      return memory.takeError();
    group.memory = std::move(*memory);
    if (matmul.getCommands().size() > static_cast<size_t>(maxScheduleCommands - total))
      return llvm::createStringError("schedule exceeds command limit");
    auto commands = decodeCommands(matmul.getCommands());
    if (!commands)
      return commands.takeError();
    total += commands->size();
    group.commands = std::move(*commands);
    input.matmuls.push_back(std::move(group));
  }
  return input;
}

DictionaryAttr mlir::npu::encodeSchedule(Builder &builder,
                                        const ExecutionSchedule &schedule) {
  SmallVector<Attribute> commands;
  commands.reserve(schedule.commands.size());
  for (const auto &record : schedule.commands) {
    SmallVector<Attribute> dependencies;
    dependencies.reserve(record.dependencies.size());
    for (int64_t dep : record.dependencies)
      dependencies.push_back(builder.getI64IntegerAttr(dep));
    commands.push_back(builder.getDictionaryAttr({
        builder.getNamedAttr("command_id", builder.getI64IntegerAttr(record.commandId)),
        builder.getNamedAttr("matmul_index", builder.getI64IntegerAttr(record.origin.matmulIndex)),
        builder.getNamedAttr("local_event_index", builder.getI64IntegerAttr(record.origin.localEventIndex)),
        builder.getNamedAttr("sequence_index", builder.getI64IntegerAttr(record.sequenceIndex)),
        builder.getNamedAttr("dependencies", builder.getArrayAttr(dependencies)),
    }));
  }
  return builder.getDictionaryAttr({
      builder.getNamedAttr("schema_version", builder.getI64IntegerAttr(1)),
      builder.getNamedAttr("contract", builder.getStringAttr(serialSRAMSchedule)),
      builder.getNamedAttr("total_commands", builder.getI64IntegerAttr(commands.size())),
      builder.getNamedAttr("commands", builder.getArrayAttr(commands)),
  });
}

llvm::Expected<ExecutionSchedule> mlir::npu::decodeSchedule(DictionaryAttr attr) {
  if (!attr || attr.size() != 4)
    return llvm::createStringError("invalid execution schedule dictionary schema");
  auto version = readIndex(attr.get("schema_version"), "schema_version");
  if (!version)
    return version.takeError();
  if (*version != 1)
    return llvm::createStringError("unsupported execution schedule schema_version");
  auto contract = dyn_cast_or_null<StringAttr>(attr.get("contract"));
  if (!contract || contract.getValue() != serialSRAMSchedule)
    return llvm::createStringError("unsupported execution schedule contract");
  auto total = readIndex(attr.get("total_commands"), "total_commands");
  if (!total)
    return total.takeError();
  auto commands = dyn_cast_or_null<ArrayAttr>(attr.get("commands"));
  if (!commands || *total > maxScheduleCommands ||
      static_cast<size_t>(*total) != commands.size())
    return llvm::createStringError("invalid execution schedule command count");

  ExecutionSchedule schedule;
  schedule.commands.reserve(commands.size());
  size_t edgeCount = 0;
  for (auto [index, entry] : llvm::enumerate(commands)) {
    auto dict = dyn_cast<DictionaryAttr>(entry);
    if (!dict || dict.size() != 5)
      return llvm::createStringError("invalid scheduled command #%zu schema", index);
    ScheduledCommand command{};
    for (auto [name, field] : {
             std::pair{"command_id", &command.commandId},
             std::pair{"matmul_index", &command.origin.matmulIndex},
             std::pair{"local_event_index", &command.origin.localEventIndex},
             std::pair{"sequence_index", &command.sequenceIndex}}) {
      auto value = readIndex(dict.get(name), name);
      if (!value)
        return value.takeError();
      *field = *value;
    }
    auto deps = dyn_cast_or_null<ArrayAttr>(dict.get("dependencies"));
    if (!deps || deps.size() > maxScheduleEdges - edgeCount)
      return llvm::createStringError("invalid dependency array on scheduled command #%zu", index);
    edgeCount += deps.size();
    for (Attribute dep : deps) {
      auto id = readIndex(dep, "dependency ID");
      if (!id)
        return id.takeError();
      command.dependencies.push_back(*id);
    }
    schedule.commands.push_back(std::move(command));
  }
  return schedule;
}
