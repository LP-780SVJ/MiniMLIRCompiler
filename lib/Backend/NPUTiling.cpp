#include "NPU/NPUTiling.h"

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

llvm::Expected<TileShape> loadNativeLimits(StringRef path) {
  auto profile = loadTargetProfile(path);
  if (!profile)
    return profile.takeError();
  return readNativeLimits(*profile);
}

struct NPUTilingPass final : PassWrapper<NPUTilingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NPUTilingPass)

  NPUTilingPass() = default;
  NPUTilingPass(const NPUTilingPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "npu-tiling"; }
  StringRef getDescription() const override {
    return "Plan static Virtual NPU MatMul tiles from a target profile";
  }

  Option<std::string> config{*this, "config",
                             llvm::cl::desc("Virtual NPU target JSON")};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NPUDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto limits = loadNativeLimits(config.getValue());
    if (!limits) {
      module.emitError() << "target config '" << config.getValue() << "': "
                         << llvm::toString(limits.takeError());
      signalPassFailure();
      return;
    }

    SmallVector<std::pair<MatMulOp, TilePlan>> work;
    WalkResult validated = module.walk([&](Operation *op) -> WalkResult {
      if (auto matmul = dyn_cast<MatMulOp>(op)) {
        auto lhs = matmul.getLhs().getType();
        auto rhs = matmul.getRhs().getType();
        auto plan = TilePlan::create(
            {lhs.getDimSize(0), rhs.getDimSize(1), lhs.getDimSize(1)}, *limits);
        if (!plan) {
          matmul.emitOpError() << llvm::toString(plan.takeError());
          return WalkResult::interrupt();
        }
        work.emplace_back(matmul, *plan);
        return WalkResult::advance();
      }
      if (auto tiled = dyn_cast<TiledMatMulOp>(op)) {
        if (tiled.getTileMAttr().getInt() == limits->m &&
            tiled.getTileNAttr().getInt() == limits->n &&
            tiled.getTileKAttr().getInt() == limits->k)
          return WalkResult::advance();
        tiled.emitOpError() << "existing tile plan does not match target config limits";
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
      op->emitOpError() << "unsupported operation in npu-tiling input";
      return WalkResult::interrupt();
    });
    if (validated.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    IRRewriter rewriter(&getContext());
    for (auto &[matmul, plan] : work) {
      SmallVector<Attribute> tiles;
      tiles.reserve(plan.size());
      for (int64_t index = 0; index < plan.size(); ++index)
        tiles.push_back(encodeTileDesc(rewriter, plan.getTile(index)));

      rewriter.setInsertionPoint(matmul);
      auto tiled = TiledMatMulOp::create(
          rewriter, matmul.getLoc(), matmul.getResult().getType(),
          matmul.getLhs(), matmul.getRhs(), virtualNPUTarget, limits->m,
          limits->n, limits->k, rewriter.getArrayAttr(tiles));
      rewriter.replaceOp(matmul, tiled.getResult());
    }
  }
};

}

void mlir::npu::registerNPUTilingPass() {
  PassRegistration<NPUTilingPass>();
}
