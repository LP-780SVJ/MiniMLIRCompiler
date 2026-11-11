#include "Mini/MiniOps.h"
#include "Mini/MiniPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <utility>

using namespace mlir;
using namespace mlir::mini;

namespace {

struct FuseAddReluPattern : OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp reluOp,
                                PatternRewriter &rewriter) const override {
    AddOp addOp = reluOp.getInput().getDefiningOp<AddOp>();
    if (!addOp)
      return rewriter.notifyMatchFailure(reluOp, "input is not mini.add");

    // The Add result is consumed only by this Relu, so removing Add is safe.
    if (!addOp.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(reluOp, "add has multiple uses");

    if (addOp.getResult().getType() != reluOp.getResult().getType())
      return rewriter.notifyMatchFailure(reluOp, "result type mismatch");

    rewriter.setInsertionPoint(reluOp);
    auto fusedOp = AddReluOp::create(
        rewriter, reluOp.getLoc(), reluOp.getResult().getType(), addOp.getLhs(),
        addOp.getRhs());
    rewriter.replaceOp(reluOp, fusedOp.getResult());
    rewriter.eraseOp(addOp);
    return success();
  }
};

struct AddReluFusionPass
    : PassWrapper<AddReluFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AddReluFusionPass)

  StringRef getArgument() const override { return "mini-fuse-add-relu"; }
  StringRef getDescription() const override {
    return "Fuse mini.add followed by mini.relu";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseAddReluPattern>(&getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (failed(applyPatternsGreedily(getOperation().getOperation(),
                                     frozenPatterns)))
      signalPassFailure();
  }
};

} // namespace

void mlir::mini::registerMiniPasses() {
  PassRegistration<AddReluFusionPass>();
}
