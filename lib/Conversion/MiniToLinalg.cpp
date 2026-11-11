#include "Mini/MiniLowering.h"
#include "Mini/MiniOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

// Both Mini operations have identical tensor access. Only the scalar body
// differs, so keep their shape and indexing rules in one conversion pattern.
template <typename MiniOp, typename ScalarOp>
struct BinaryOpLowering : OpConversionPattern<MiniOp> {
  using OpConversionPattern<MiniOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<MiniOp>::OpAdaptor;

  LogicalResult matchAndRewrite(MiniOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (lhs.getType() != resultType || rhs.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operands and result must have identical tensor types");

    Location loc = op.getLoc();
    auto empty = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        resultType.getEncoding());

    unsigned rank = resultType.getRank();
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps{identityMap, identityMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultType, ValueRange{lhs, rhs},
        ValueRange{empty.getResult()}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location nestedLoc, ValueRange args) {
          Value scalarResult =
              ScalarOp::create(builder, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(builder, nestedLoc, scalarResult);
        });
    rewriter.replaceOp(op, generic->getResults());
    return success();
  }
};

using AddOpLowering = BinaryOpLowering<mini::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<mini::MulOp, arith::MulFOp>;

struct MiniToLinalgPass
    : PassWrapper<MiniToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MiniToLinalgPass)

  StringRef getArgument() const override { return "mini-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower mini.add and mini.mul to tensor/linalg/arith";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext &context = getContext();
    ConversionTarget target(context);
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect, func::FuncDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<mini::MiniDialect>();
    target.addLegalOp<mini::ReluOp, mini::AddReluOp>();

    RewritePatternSet patterns(&context);
    patterns.add<AddOpLowering, MulOpLowering>(&context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::mini::registerMiniLoweringPasses() {
  PassRegistration<MiniToLinalgPass>();
}
