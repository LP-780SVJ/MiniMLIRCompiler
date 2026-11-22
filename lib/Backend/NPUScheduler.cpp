#include "NPU/NPUScheduler.h"

#include "NPU/NPUDialect.h"
#include "NPU/SchedulePlanAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::npu;

namespace {

struct NPUSchedulerPass final
    : PassWrapper<NPUSchedulerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NPUSchedulerPass)

  StringRef getArgument() const override { return "npu-schedule"; }
  StringRef getDescription() const override {
    return "Build and verify a global serial NPU command schedule";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NPUDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    if (auto error = scheduleModule(module)) {
      module.emitError("NPU schedule: ") << llvm::toString(std::move(error));
      signalPassFailure();
    }
  }

  llvm::Error scheduleModule(ModuleOp module) {
    auto input = readScheduleInput(module);
    if (!input)
      return input.takeError();
    Attribute existing = module->getAttr(executionScheduleAttr);
    auto schedule = existing ? decodeSchedule(dyn_cast<DictionaryAttr>(existing))
                             : buildSchedule(input->layout, input->matmuls);
    if (!schedule)
      return schedule.takeError();
    if (auto error = verifySchedule(input->layout, input->matmuls, *schedule))
      return error;
    if (!existing) {
      Builder builder(module.getContext());
      module->setAttr(executionScheduleAttr, encodeSchedule(builder, *schedule));
    }
    return llvm::Error::success();
  }
};

}

void mlir::npu::registerNPUSchedulerPass() {
  PassRegistration<NPUSchedulerPass>();
}
