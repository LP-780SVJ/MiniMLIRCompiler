#include "NPU/NPUDialect.h"
#include "NPU/CommandPlanAttrs.h"
#include "NPU/NPUOps.h"
#include "NPU/SchedulePlanAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include <utility>

#include "NPU/NPUOpsDialect.cpp.inc"

void mlir::npu::NPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NPU/NPUOps.cpp.inc"
      >();
}

mlir::LogicalResult
mlir::npu::NPUDialect::verifyOperationAttribute(Operation *op,
                                               NamedAttribute attr) {
  if (attr.getName().strref() == executionScheduleAttr) {
    auto module = dyn_cast<ModuleOp>(op);
    if (!module)
      return op->emitOpError("requires npu.execution_schedule on a module");
    auto schedule = decodeSchedule(dyn_cast<DictionaryAttr>(attr.getValue()));
    if (!schedule)
      return op->emitOpError("NPU schedule: ") << llvm::toString(schedule.takeError());
    auto input = readScheduleInput(module);
    if (!input)
      return op->emitOpError("NPU schedule: ") << llvm::toString(input.takeError());
    if (auto error = verifySchedule(input->layout, input->matmuls, *schedule))
      return op->emitOpError("NPU schedule: ") << llvm::toString(std::move(error));
    return success();
  }
  if (attr.getName().strref() != dramLayoutAttr)
    return success();
  auto module = dyn_cast<ModuleOp>(op);
  auto dict = dyn_cast<DictionaryAttr>(attr.getValue());
  if (!module || !dict)
    return op->emitOpError("requires npu.dram_layout to be a module dictionary");

  auto supplied = decodeDRAMLayout(dict);
  if (!supplied)
    return op->emitOpError() << llvm::toString(supplied.takeError());
  llvm::DenseMap<Value, int64_t> ids;
  auto expected = buildDRAMLayout(module, DRAMLayoutStage::Commands, ids);
  if (!expected)
    return op->emitOpError() << llvm::toString(expected.takeError());
  Builder builder(op->getContext());
  if (encodeDRAMLayout(builder, *expected) != dict)
    return op->emitOpError("DRAM layout does not match the SSA tensor layout or ordered I/O");

  Operation *previous = nullptr;
  for (Operation &command : module.getBody()->front().getRegion(0).front()) {
    if (!isa<CommandMatMulOp>(command))
      continue;
    if (previous) {
      for (StringRef field : {"target", "tile_m", "tile_n", "tile_k"}) {
        if (previous->getAttr(field) != command.getAttr(field))
          return command.emitOpError("inconsistent target across command MatMuls");
      }
      auto prior = previous->getAttrOfType<DictionaryAttr>("memory_plan");
      auto current = command.getAttrOfType<DictionaryAttr>("memory_plan");
      if (!prior || !current)
        return command.emitOpError("expected memory_plan on every command MatMul");
      for (StringRef field : {"sram_capacity_bytes", "alignment_bytes",
                              "reserved_bytes", "schedule_contract"}) {
        if (prior.get(field) != current.get(field))
          return command.emitOpError("inconsistent SRAM profile across command MatMuls");
      }
    }
    previous = &command;
    for (auto [name, value] : {std::pair{"lhs_tensor_id", command.getOperand(0)},
                              std::pair{"rhs_tensor_id", command.getOperand(1)},
                              std::pair{"result_tensor_id", Value(command.getResult(0))}}) {
      auto id = command.getAttrOfType<IntegerAttr>(name);
      auto expectedId = ids.find(value);
      if (!id || !id.getType().isSignlessInteger(64) || expectedId == ids.end() ||
          id.getInt() != expectedId->second)
        return command.emitOpError() << name << " does not match SSA tensor binding";
    }
  }
  return success();
}
