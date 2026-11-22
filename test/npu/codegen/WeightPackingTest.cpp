#include "NPU/ProgramPackage.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Endian.h"

using namespace mlir;
using namespace mlir::npu;

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect>();
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @main(%x: tensor<1x1xf32>) {
      %words = arith.constant dense<[[0x00000000, 0x80000000, 0x00000001, 0x80000001],
                                    [0x7FC12345, 0x7F800000, 0xFF800000, 0x7F812345]]>
                                  : tensor<2x4xf32>
      %splat = arith.constant dense<0x80000000> : tensor<1x2xf32>
      return
    }
  )mlir", &context);
  if (!module)
    return 1;
  DRAMLayout layout{64, 136,
                    {{0, TensorStorageKind::Input, 0, 0, 1, 1, 4, 4},
                     {1, TensorStorageKind::Constant, 0, 64, 2, 4, 16, 32},
                     {2, TensorStorageKind::Constant, 1, 128, 1, 2, 8, 8}},
                    {0}, {1, 2}, {}};
  auto packed = packWeights(*module, layout);
  if (!packed) {
    llvm::logAllUnhandledErrors(packed.takeError(), llvm::errs());
    return 1;
  }
  const uint32_t words[] = {0x00000000, 0x80000000, 0x00000001, 0x80000001,
                            0x7FC12345, 0x7F800000, 0xFF800000, 0x7F812345};
  StringRef bytes(packed->bytes);
  if (bytes.size() != 72 || bytes.slice(32, 64) != std::string(32, '\0') ||
      packed->entries.size() != 2)
    return 1;
  for (auto [i, word] : llvm::enumerate(words))
    if (llvm::support::endian::read32le(bytes.data() + 4 * i) != word)
      return 1;
  for (unsigned offset : {64, 68})
    if (llvm::support::endian::read32le(bytes.data() + offset) != 0x80000000)
      return 1;
  auto *second = packed->entries[1].getAsObject();
  if (!second || second->getInteger("tensor_id") != 2 ||
      second->getInteger("file_offset") != 64 || second->getInteger("dram_base") != 128 ||
      second->getInteger("size_bytes") != 8)
    return 1;

  layout.tensors[2].sizeBytes = 4;
  if (auto invalid = packWeights(*module, layout)) {
    llvm::errs() << "packed a constant with an inconsistent descriptor\n";
    return 1;
  } else {
    llvm::consumeError(invalid.takeError());
  }
  return 0;
}
