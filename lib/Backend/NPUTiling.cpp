#include "NPU/NPUTiling.h"

#include "NPU/NPUOps.h"
#include "NPU/TilePlanAttrs.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <limits>
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::npu;

namespace {

llvm::Expected<TileShape> loadNativeLimits(StringRef path) {
  if (path.empty())
    return llvm::createStringError("npu-tiling requires a config path");
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError("cannot read target config: %s",
                                   buffer.getError().message().c_str());
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return parsed.takeError();
  auto *profile = parsed->getAsObject();
  const auto *schema = profile ? profile->get("schema_version") : nullptr;
  if (!schema || schema->getAsUINT64() != 1)
    return llvm::createStringError("expected target schema_version 1");
  if (profile->getString("target") != virtualNPUTarget)
    return llvm::createStringError("expected target virtual-npu-v1");
  const auto *compute = profile->getObject("compute");
  if (!compute)
    return llvm::createStringError("missing compute object");
  const auto *units = compute->get("matrix_units");
  if (!units || units->getAsUINT64() != 1)
    return llvm::createStringError("expected one matrix unit");
  if (compute->getString("input_dtype") != "float32" ||
      compute->getString("accumulator_dtype") != "float32")
    return llvm::createStringError("expected float32 input and accumulator");
  if (compute->getString("matmul_semantics") != "accumulate")
    return llvm::createStringError("expected accumulate MatMul semantics");

  const auto *native = compute->getObject("native_matmul_tile");
  if (!native)
    return llvm::createStringError("missing compute.native_matmul_tile object");
  TileShape limits{};
  for (auto [name, limit] : {std::pair{"m", &limits.m}, std::pair{"n", &limits.n},
                             std::pair{"k", &limits.k}}) {
    const auto *value = native->get(name);
    auto size = value ? value->getAsUINT64() : std::nullopt;
    if (!size || *size == 0 ||
        *size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return llvm::createStringError(
          "native_matmul_tile.%s must be a positive i64 integer", name);
    *limit = static_cast<int64_t>(*size);
  }
  return limits;
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
