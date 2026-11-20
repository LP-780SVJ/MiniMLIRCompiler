#include "Mini/MiniNPULowering.h"

#include "Mini/MiniOps.h"
#include "NPU/NPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

struct MatMulLowering : OpConversionPattern<mini::MatMulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mini::MatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto lowered = npu::MatMulOp::create(rewriter, op.getLoc(),
                                         op.getResult().getType(),
                                         adaptor.getLhs(), adaptor.getRhs());
    rewriter.replaceOp(op, lowered.getResult());
    return success();
  }
};

struct MiniToNPUPass : PassWrapper<MiniToNPUPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MiniToNPUPass)

  StringRef getArgument() const override { return "mini-to-npu"; }
  StringRef getDescription() const override {
    return "Fully lower supported Mini operations to Virtual NPU compute IR";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<npu::NPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, npu::MatMulOp>();
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
      auto type = dyn_cast<RankedTensorType>(op.getType());
      return type && type.hasStaticShape() && type.getElementType().isF32() &&
             isa<DenseFPElementsAttr>(op.getValue());
    });
    target.addIllegalDialect<mini::MiniDialect>();

    RewritePatternSet patterns(&context);
    patterns.add<MatMulLowering>(&context);
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

}

std::unique_ptr<Pass> mlir::mini::createMiniToNPUPass() {
  return std::make_unique<MiniToNPUPass>();
}

void mlir::mini::registerMiniToNPUPass() {
  PassRegistration<MiniToNPUPass>();
}
