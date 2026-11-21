#include "NPU/CommandPlan.h"

#include "llvm/Support/raw_ostream.h"
#include <limits>

using namespace mlir::npu;

namespace {

bool expectEnd(llvm::Expected<int64_t> result, int64_t expected) {
  if (!result) {
    llvm::errs() << llvm::toString(result.takeError()) << "\n";
    return false;
  }
  if (*result == expected)
    return true;
  llvm::errs() << "Expected DMA end " << expected << ", got " << *result
               << "\n";
  return false;
}

bool expectError(llvm::Expected<int64_t> result, llvm::StringRef expected) {
  if (result) {
    llvm::errs() << "Expected error, got DMA end " << *result << "\n";
    return false;
  }
  auto message = llvm::toString(result.takeError());
  llvm::outs() << message << "\n";
  return llvm::StringRef(message).contains(expected);
}

}

int main() {
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  const char *badGeometry = "invalid 2D DMA dimensions or stride";
  const char *overflow = "DMA byte range overflow";
  return expectEnd(checkedDMAEnd(12544, 32, 128, 384), 24576) &&
                 expectEnd(checkedDMAEnd(4096, 32, 128, 128), 8192) &&
                 expectEnd(checkedDMAEnd(max - 16, 1, 16, max), max) &&
                 expectError(checkedDMAEnd(0, 0, 32, 32), badGeometry) &&
                 expectError(checkedDMAEnd(0, 1, 0, 32), badGeometry) &&
                 expectError(checkedDMAEnd(0, 1, 32, 16), badGeometry) &&
                 expectError(checkedDMAEnd(-1, 1, 32, 32), badGeometry) &&
                 expectError(checkedDMAEnd(0, 3, 16, max), overflow) &&
                 expectError(checkedDMAEnd(16, 2, 16, max), overflow) &&
                 expectError(checkedDMAEnd(max - 15, 1, 16, 16), overflow)
             ? 0
             : 1;
}
