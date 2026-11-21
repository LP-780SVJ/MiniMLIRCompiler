#include "NPU/MemoryPlan.h"

#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <string>

using namespace mlir::npu;

namespace {

template <typename Value>
bool expectError(llvm::Expected<Value> result, const char *diagnostic) {
  if (result) {
    llvm::errs() << "Expected error: " << diagnostic << "\n";
    return false;
  }
  std::string message = llvm::toString(result.takeError());
  llvm::outs() << message << "\n";
  return message.find(diagnostic) != std::string::npos;
}

bool checkAddresses() {
  const BufferRequirement requests[] = {
      {0, BufferRole::C, 0, 0, 64, 64, 64, 0, 10},
      {1, BufferRole::A, 0, 0, 32, 64, 64, 0, 1},
      {2, BufferRole::B, 0, 0, 63, 64, 64, 1, 2},
  };
  auto allocations = allocateBuffers(requests, {192, 1, 64});
  if (!allocations) {
    llvm::errs() << llvm::toString(allocations.takeError()) << "\n";
    return false;
  }
  const auto &placed = *allocations;
  bool valid = placed.size() == 3 && placed[0].address == 64 &&
               placed[1].address == 128 && placed[2].address == 128 &&
               !placed[1].reusedAddress && placed[2].reusedAddress;
  if (!valid)
    llvm::errs() << "aligned reserve or half-open lifetime reuse failed\n";
  return valid;
}

}

int main() {
  const BufferRequirement fragmented[] = {
      {0, BufferRole::A, 0, 0, 64, 64, 64, 0, 10},
      {1, BufferRole::A, 1, 0, 64, 64, 64, 0, 1},
      {2, BufferRole::A, 2, 0, 64, 64, 64, 0, 10},
      {3, BufferRole::A, 3, 0, 64, 64, 64, 0, 1},
      {4, BufferRole::A, 4, 0, 128, 128, 64, 1, 2},
  };
  if (!expectError(allocateBuffers(fragmented, {256, 0, 64}),
                   "First-Fit fragmentation") ||
      !expectError(allocateBuffers(fragmented, {192, 0, 64}),
                   "insufficient SRAM capacity"))
    return 1;

  constexpr auto largest = std::numeric_limits<int64_t>::max();
  const TileDesc huge[] = {{0, 0, 0, largest, 32, 32, 0, true, true}};
  if (!expectError(generateMemoryPlan(huge, {262144, 0, 64}),
                   "FP32 buffer byte size overflow"))
    return 1;
  const TileDesc macro[] = {{0, 0, 0, 256, 256, 256, 0, true, true}};
  if (!expectError(generateMemoryPlan(macro, {262144, 0, 64}),
                   "insufficient SRAM capacity"))
    return 1;
  return checkAddresses() ? 0 : 1;
}
