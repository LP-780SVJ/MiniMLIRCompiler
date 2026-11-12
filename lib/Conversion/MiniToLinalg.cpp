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

// ReLU(x) = x for ordered x > +0.0, otherwise +0.0. This chooses +0.0
// for NaN and -0.0; both Mini lowering paths must share the same rule.
static Value emitReluScalar(OpBuilder &builder, Location loc, Value input) {
  auto zero = arith::ConstantOp::create(
      builder, loc, builder.getF32FloatAttr(0.0f));
  auto positive = arith::CmpFOp::create(
      builder, loc, arith::CmpFPredicate::OGT, input, zero.getResult());
  auto result = arith::SelectOp::create(
      builder, loc, positive.getResult(), input, zero.getResult());
  return result.getResult();
}

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

struct ReluOpLowering : OpConversionPattern<mini::ReluOp> {
  using OpConversionPattern<mini::ReluOp>::OpConversionPattern;
  using OpAdaptor = OpConversionPattern<mini::ReluOp>::OpAdaptor;

  LogicalResult matchAndRewrite(mini::ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    Value input = adaptor.getInput();
    if (input.getType() != resultType)
      return rewriter.notifyMatchFailure(op, "input and result types must match");

    Location loc = op.getLoc();
    auto empty = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        resultType.getEncoding());
    unsigned rank = resultType.getRank();
    AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
    SmallVector<AffineMap> indexingMaps{identityMap, identityMap};
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, resultType, ValueRange{input},
        ValueRange{empty.getResult()}, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location nestedLoc, ValueRange args) {
          Value activated = emitReluScalar(builder, nestedLoc, args[0]);
          linalg::YieldOp::create(builder, nestedLoc, activated);
        });
    rewriter.replaceOp(op, generic->getResults());
    return success();
  }
};

struct AddReluOpLowering : OpConversionPattern<mini::AddReluOp> {
  using OpConversionPattern<mini::AddReluOp>::OpConversionPattern;
  using OpAdaptor = OpConversionPattern<mini::AddReluOp>::OpAdaptor;

  LogicalResult matchAndRewrite(mini::AddReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (lhs.getType() != resultType || rhs.getType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "operands and result types must match");

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
          auto sum = arith::AddFOp::create(
              builder, nestedLoc, args[0], args[1]);
          Value activated = emitReluScalar(builder, nestedLoc, sum.getResult());
          linalg::YieldOp::create(builder, nestedLoc, activated);
        });
    rewriter.replaceOp(op, generic->getResults());
    return success();
  }
};

struct MiniToLinalgPass
    : PassWrapper<MiniToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MiniToLinalgPass)

  StringRef getArgument() const override { return "mini-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower Mini tensor operations to tensor/linalg/arith";
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

    RewritePatternSet patterns(&context);
    patterns.add<AddOpLowering, MulOpLowering, ReluOpLowering,
                 AddReluOpLowering>(&context);
    if (failed(applyFullConversion(getOperation(), target,
                                   std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::mini::registerMiniLoweringPasses() {
  PassRegistration<MiniToLinalgPass>();
}

std::unique_ptr<mlir::Pass> mlir::mini::createMiniToLinalgPass() {
  return std::make_unique<MiniToLinalgPass>();
}
