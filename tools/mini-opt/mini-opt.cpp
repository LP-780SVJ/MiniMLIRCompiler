
#include "Mini/MiniDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // 注册本项目的 Mini Dialect
  registry.insert<mlir::mini::MiniDialect>();

  // 注册本周示例可能使用的标准 Dialect
  registry.insert<
      mlir::arith::ArithDialect,
      mlir::func::FuncDialect>();

  // 复用 MLIR 的 opt 工具基础设施
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(
          argc,
          argv,
          "MiniMLIRCompiler optimizer\n",
          registry));
}
