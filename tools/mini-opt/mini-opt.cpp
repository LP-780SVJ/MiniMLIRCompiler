
#include "Mini/MiniDialect.h"
#include "Mini/MiniLowering.h"
#include "Mini/MiniPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::mini::registerMiniPasses();
  mlir::mini::registerMiniLoweringPasses();
  mlir::registerTransformsPasses();
  mlir::PassPipelineRegistration<>(
      "mini-compile", "Canonicalize, fuse and fully lower Mini IR",
      [](mlir::OpPassManager &pm) {
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::mini::createAddReluFusionPass());
        pm.addPass(mlir::mini::createMiniToLinalgPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
      });
  mlir::DialectRegistry registry;

  // 注册本项目的 Mini Dialect
  registry.insert<mlir::mini::MiniDialect>();

  // 注册本周示例可能使用的标准 Dialect
  registry.insert<
      mlir::arith::ArithDialect,
      mlir::func::FuncDialect,
      mlir::linalg::LinalgDialect,
      mlir::tensor::TensorDialect>();

  // 复用 MLIR 的 opt 工具基础设施
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(
          argc,
          argv,
          "MiniMLIRCompiler optimizer\n",
          registry));
}
