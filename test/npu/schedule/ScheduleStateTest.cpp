#include "NPU/SchedulePlan.h"
#include "llvm/Support/raw_ostream.h"

#include <tuple>
#include <utility>

using namespace mlir::npu;

namespace {

DRAMTensorDesc tensor(int64_t id, TensorStorageKind kind, int64_t ordinal,
                       int64_t base, int64_t rows, int64_t cols) {
  return {id, kind, ordinal, base, rows, cols, cols * 4, rows * cols * 4};
}

ScheduleMatMul makeMatMul(const DRAMLayout &layout,
                          std::array<int64_t, 3> ids) {
  auto lhs = layout.tensors[ids[0]];
  auto rhs = layout.tensors[ids[1]];
  TilePlan plan = llvm::cantFail(
      TilePlan::create({lhs.rows, rhs.cols, lhs.cols}, {32, 32, 32}));
  std::vector<TileDesc> tiles;
  for (int64_t index = 0; index < plan.size(); ++index)
    tiles.push_back(plan.getTile(index));
  auto memory = llvm::cantFail(generateMemoryPlan(tiles, {262144, 0, 64}));
  auto commands = llvm::cantFail(
      generateCommands(tiles, memory, {lhs, rhs, layout.tensors[ids[2]]}));
  return {ids, lhs.cols, std::move(tiles), std::move(memory), std::move(commands)};
}

std::vector<ScheduledCommand> serial(llvm::ArrayRef<ScheduleMatMul> matmuls) {
  std::vector<ScheduledCommand> entries;
  for (size_t index = 0; index < matmuls.size(); ++index)
    for (const auto &command : matmuls[index].commands) {
      auto id = static_cast<int64_t>(entries.size());
      entries.push_back({id, {static_cast<int64_t>(index), command.eventIndex},
                         id, {}});
    }
  return entries;
}

bool rejects(const char *label, const DRAMLayout &layout,
             llvm::ArrayRef<ScheduleMatMul> matmuls,
             llvm::ArrayRef<ScheduledCommand> order, llvm::StringRef diagnostic) {
  auto error = verifyScheduleState(layout, matmuls, order);
  if (!error) {
    llvm::errs() << label << ": unexpectedly accepted\n";
    return false;
  }
  auto message = llvm::toString(std::move(error));
  llvm::outs() << label << ": " << message << '\n';
  return llvm::StringRef(message).contains(diagnostic);
}

bool accepts(const DRAMLayout &layout, llvm::ArrayRef<ScheduleMatMul> matmuls,
             llvm::ArrayRef<ScheduledCommand> order) {
  if (auto error = verifyScheduleState(layout, matmuls, order)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return false;
  }
  return true;
}

bool checkSingle() {
  DRAMLayout layout{64, 20736,
                    {tensor(0, TensorStorageKind::Input, 0, 0, 32, 65),
                     tensor(1, TensorStorageKind::Input, 1, 8320, 65, 32),
                     tensor(2, TensorStorageKind::Output, 0, 16640, 32, 32)},
                    {0, 1}, {}, {2}};
  auto base = makeMatMul(layout, {0, 1, 2});
  auto order = serial(base);
  if (!accepts(layout, base, order))
    return false;

  for (auto [label, first, second, diagnostic] : {
           std::tuple{"compute before load", 2, 3, "MATMUL reads A/B before LOAD"},
           std::tuple{"compute before zero", 0, 3, "MATMUL reads C before ZERO"},
           std::tuple{"premature reuse", 3, 4, "SRAM write overlaps a still-live buffer"},
           std::tuple{"premature store", 9, 10, "STORE before the K reduction is complete"},
           std::tuple{"wrong K order", 3, 6, "MATMUL violates K reduction order"},
       }) {
    auto altered = order;
    std::swap(altered[first].origin, altered[second].origin);
    if (!rejects(label, layout, base, altered, diagnostic))
      return false;
  }

  auto repeatedZero = base;
  repeatedZero.commands[4].payload = base.commands[0].payload;
  if (!rejects("repeated initialization", layout, repeatedZero, order,
               "repeated ZERO"))
    return false;

  auto earlyRelease = base;
  earlyRelease.memory.allocations[0].requirement.liveEnd = 4;
  if (!rejects("early C release", layout, earlyRelease, order,
               "actual SRAM lifetime does not match the Memory Plan"))
    return false;

  auto overlap = base;
  auto &loadB = std::get<DMALoad2DCommand>(overlap.commands[2].payload);
  loadB.dstSram = overlap.memory.allocations[1].address;
  overlap.memory.allocations[2].address = loadB.dstSram;
  if (!rejects("simultaneously live SRAM allocations", layout, overlap, order,
               "SRAM write overlaps a still-live buffer"))
    return false;

  order.pop_back();
  return rejects("missing last store", layout, base, order, "missing STORE regions");
}

bool checkPartialDRAM() {
  DRAMLayout layout{64, 1608,
                    {tensor(0, TensorStorageKind::Input, 0, 0, 2, 3),
                     tensor(1, TensorStorageKind::Input, 1, 64, 3, 64),
                     tensor(2, TensorStorageKind::Input, 2, 832, 64, 1),
                     tensor(3, TensorStorageKind::Intermediate, 0, 1088, 2, 64),
                     tensor(4, TensorStorageKind::Output, 1, 1600, 2, 1)},
                    {0, 1, 2}, {}, {4}};
  std::vector<ScheduleMatMul> chain{makeMatMul(layout, {0, 1, 3}),
                                   makeMatMul(layout, {3, 2, 4})};
  auto order = serial(chain);
  if (!accepts(layout, chain, order))
    return false;
  size_t loadPosition = chain.front().commands.size() + 1;
  auto earlyLoad = order[loadPosition].origin;
  order.resize(5);
  order.push_back({static_cast<int64_t>(loadPosition), earlyLoad, 5, {}});
  return rejects("read after only one of two STORE regions", layout, chain,
                 order, "LOAD reads a DRAM Tensor before all output tiles are written");
}

}

int main() {
  return checkSingle() && checkPartialDRAM() ? 0 : 1;
}
