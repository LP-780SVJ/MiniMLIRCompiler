#ifndef NPU_SCHEDULEPLANATTRS_H
#define NPU_SCHEDULEPLANATTRS_H

#include "NPU/SchedulePlan.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
class Builder;

namespace npu {

inline constexpr char executionScheduleAttr[] = "npu.execution_schedule";

struct ScheduleInput {
  DRAMLayout layout;
  std::vector<ScheduleMatMul> matmuls;
};

llvm::Expected<ScheduleInput> readScheduleInput(ModuleOp module);
DictionaryAttr encodeSchedule(Builder &builder, const ExecutionSchedule &schedule);
llvm::Expected<ExecutionSchedule> decodeSchedule(DictionaryAttr attr);

}
}

#endif
