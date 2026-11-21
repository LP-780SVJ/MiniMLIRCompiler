#include "NPU/SRAMPlanner.h"

#include "NPU/MemoryPlanAttrs.h"
#include "NPU/NPUOps.h"
#include "NPU/TilePlanAttrs.h"
#include "NPU/VirtualNPUProfile.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <utility>

using namespace mlir;
using namespace mlir::npu;

namespace {

struct Profile {
  TileShape native;
  SRAMConfig memory;
};

llvm::Expected<Profile> loadProfile(StringRef path) {
  auto profile = loadTargetProfile(path);
  if (!profile)
    return profile.takeError();
  auto native = readNativeLimits(*profile);
  if (!native)
    return native.takeError();

  auto memory = readSRAMConfig(*profile);
  if (!memory)
    return memory.takeError();
  return Profile{*native, *memory};
}

template <typename MatMul>
bool matchesNative(MatMul op, TileShape limits) {
  return op.getTarget() == virtualNPUTarget &&
         op.getTileMAttr().getInt() == limits.m &&
         op.getTileNAttr().getInt() == limits.n &&
         op.getTileKAttr().getInt() == limits.k;
}

bool matchesMemory(SRAMConfig lhs, SRAMConfig rhs) {
  return lhs.capacityBytes == rhs.capacityBytes &&
         lhs.reservedBytes == rhs.reservedBytes &&
         lhs.alignmentBytes == rhs.alignmentBytes;
}

llvm::Expected<MemoryPlan> planOperation(TiledMatMulOp op,
                                        const Profile &profile) {
  if (!matchesNative(op, profile.native))
    return llvm::createStringError("tile plan does not match target config");
  SmallVector<TileDesc> tiles;
  tiles.reserve(op.getTiles().size());
  for (Attribute attr : op.getTiles()) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict)
      return llvm::createStringError("tile must be a dictionary");
    auto tile = decodeTileDesc(dict);
    if (!tile)
      return tile.takeError();
    tiles.push_back(*tile);
  }
  auto plan = generateMemoryPlan(tiles, profile.memory);
  if (!plan)
    return plan.takeError();
  if (auto error = verifyMemoryPlan(tiles, *plan))
    return error;
  return plan;
}

struct SRAMPlannerPass final
    : PassWrapper<SRAMPlannerPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SRAMPlannerPass)

  SRAMPlannerPass() = default;
  SRAMPlannerPass(const SRAMPlannerPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "npu-sram-plan"; }
  StringRef getDescription() const override {
    return "Plan single-buffer SRAM allocations for tiled Virtual NPU MatMul";
  }

  Option<std::string> config{*this, "config",
                             llvm::cl::desc("Virtual NPU target JSON")};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto profile = loadProfile(config.getValue());
    if (!profile) {
      module.emitError() << "target config '" << config.getValue() << "': "
                         << llvm::toString(profile.takeError());
      signalPassFailure();
      return;
    }

    SmallVector<std::pair<TiledMatMulOp, MemoryPlan>> work;
    WalkResult validated = module.walk([&](Operation *op) -> WalkResult {
      if (auto tiled = dyn_cast<TiledMatMulOp>(op)) {
        auto plan = planOperation(tiled, *profile);
        if (!plan) {
          tiled.emitOpError() << llvm::toString(plan.takeError());
          return WalkResult::interrupt();
        }
        work.emplace_back(tiled, std::move(*plan));
        return WalkResult::advance();
      }
      if (auto planned = dyn_cast<SRAMPlannedMatMulOp>(op)) {
        auto plan = decodeMemoryPlan(planned.getMemoryPlan());
        if (!plan) {
          planned.emitOpError() << llvm::toString(plan.takeError());
          return WalkResult::interrupt();
        }
        if (matchesNative(planned, profile->native) &&
            matchesMemory(plan->config, profile->memory))
          return WalkResult::advance();
        planned.emitOpError() << "existing SRAM plan does not match target config";
        return WalkResult::interrupt();
      }
      if (isa<ModuleOp, func::FuncOp, func::ReturnOp>(op))
        return WalkResult::advance();
      if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        auto type = dyn_cast<RankedTensorType>(constant.getType());
        if (type && type.hasStaticShape() && type.getElementType().isF32() &&
            isa<DenseFPElementsAttr>(constant.getValue()))
          return WalkResult::advance();
      }
      if (isa<MatMulOp>(op))
        op->emitOpError() << "requires npu.tiled_matmul before SRAM planning";
      else
        op->emitOpError() << "unsupported operation in npu-sram-plan input";
      return WalkResult::interrupt();
    });
    if (validated.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    for (auto &[tiled, plan] : work) {
      rewriter.setInsertionPoint(tiled);
      auto planned = SRAMPlannedMatMulOp::create(
          rewriter, tiled.getLoc(), tiled.getResult().getType(),
          tiled.getLhs(), tiled.getRhs(), tiled.getTargetAttr(),
          tiled.getTileMAttr(), tiled.getTileNAttr(), tiled.getTileKAttr(),
          tiled.getTiles(), encodeMemoryPlan(rewriter, plan));
      rewriter.replaceOp(tiled, planned.getResult());
    }
  }
};

}

void mlir::npu::registerSRAMPlannerPass() {
  PassRegistration<SRAMPlannerPass>();
}
