#include "NPU/DMAInsertion.h"

#include "NPU/CommandPlanAttrs.h"
#include "NPU/MemoryPlanAttrs.h"
#include "NPU/NPUOps.h"
#include "NPU/TilePlanAttrs.h"
#include "NPU/VirtualNPUProfile.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::npu;

namespace {

template <typename MatMul>
bool matchesProfile(MatMul op, TileShape native, SRAMConfig config) {
  auto memory = op.getMemoryPlan();
  auto matches = [&](StringRef name, int64_t value) {
    auto field = dyn_cast_or_null<IntegerAttr>(memory.get(name));
    return field && field.getType().isSignlessInteger(64) &&
           field.getInt() == value;
  };
  return op.getTarget() == virtualNPUTarget &&
         op.getTileMAttr().getInt() == native.m &&
         op.getTileNAttr().getInt() == native.n &&
         op.getTileKAttr().getInt() == native.k &&
         matches("sram_capacity_bytes", config.capacityBytes) &&
         matches("reserved_bytes", config.reservedBytes) &&
         matches("alignment_bytes", config.alignmentBytes);
}

llvm::Expected<std::vector<CommandRecord>>
planCommands(SRAMPlannedMatMulOp op, const MatMulTensors &tensors) {
  auto memory = decodeMemoryPlan(op.getMemoryPlan());
  if (!memory)
    return memory.takeError();
  SmallVector<TileDesc> tiles;
  tiles.reserve(op.getTiles().size());
  for (Attribute entry : op.getTiles()) {
    auto dict = dyn_cast<DictionaryAttr>(entry);
    if (!dict)
      return llvm::createStringError("expected Tile descriptor dictionary");
    auto desc = decodeTileDesc(dict);
    if (!desc)
      return desc.takeError();
    tiles.push_back(*desc);
  }
  auto commands = generateCommands(tiles, *memory, tensors);
  if (!commands)
    return commands.takeError();
  if (auto error = verifyCommands(tiles, *memory, tensors, *commands))
    return error;
  return commands;
}

struct WorkItem {
  SRAMPlannedMatMulOp op;
  std::array<int64_t, 3> ids;
  std::vector<CommandRecord> commands;
};

struct DMAInsertionPass final
    : PassWrapper<DMAInsertionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DMAInsertionPass)

  DMAInsertionPass() = default;
  DMAInsertionPass(const DMAInsertionPass &other) : PassWrapper(other) {}

  StringRef getArgument() const override { return "npu-insert-dma"; }
  StringRef getDescription() const override {
    return "Lay out DRAM tensors and lower planned NPU MatMuls to DMA commands";
  }

  Option<std::string> config{*this, "config",
                             llvm::cl::desc("Virtual NPU target JSON")};

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<NPUDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    if (auto error = lowerModule(module)) {
      module.emitError() << llvm::toString(std::move(error));
      signalPassFailure();
    }
  }

  llvm::Error lowerModule(ModuleOp module) {
    auto profile = loadTargetProfile(config.getValue());
    if (!profile)
      return profile.takeError();
    auto native = readNativeLimits(*profile);
    if (!native)
      return native.takeError();
    auto sram = readSRAMConfig(*profile);
    if (!sram)
      return sram.takeError();
    if (profile->getObject("memory")->getString("dram_model") != "linear_bytes" ||
        profile->getObject("dma")->getBoolean("supports_2d_strided") != true)
      return llvm::createStringError("expected linear byte DRAM and 2D strided DMA");

    bool hasLayout = module->hasAttr(dramLayoutAttr);
    llvm::DenseMap<Value, int64_t> valueIds;
    auto layout = buildDRAMLayout(module, hasLayout ? DRAMLayoutStage::Commands
                                                    : DRAMLayoutStage::SRAMPlanned,
                                   valueIds);
    if (!layout)
      return layout.takeError();

    auto &block = module.getBody()->front().getRegion(0).front();
    SmallVector<WorkItem> work;
    for (Operation &op : block) {
      if (auto command = dyn_cast<CommandMatMulOp>(op)) {
        if (!matchesProfile(command, *native, *sram))
          return llvm::createStringError("existing command plan does not match target profile");
      } else if (auto planned = dyn_cast<SRAMPlannedMatMulOp>(op)) {
        if (!matchesProfile(planned, *native, *sram))
          return llvm::createStringError("SRAM plan does not match target profile");
        auto lhs = valueIds.find(planned.getLhs());
        auto rhs = valueIds.find(planned.getRhs());
        auto result = valueIds.find(planned.getResult());
        if (lhs == valueIds.end() || rhs == valueIds.end() ||
            result == valueIds.end())
          return llvm::createStringError("unmapped MatMul Tensor SSA value");
        std::array<int64_t, 3> ids{lhs->second, rhs->second, result->second};
        MatMulTensors tensors{layout->tensors[ids[0]], layout->tensors[ids[1]],
                              layout->tensors[ids[2]]};
        auto commands = planCommands(planned, tensors);
        if (!commands) {
          auto message = llvm::toString(commands.takeError());
          return llvm::createStringError(llvm::Twine("MatMul result tensor ") +
                                         llvm::Twine(ids[2]) + ": " + message);
        }
        work.push_back({planned, ids, std::move(*commands)});
      }
    }
    if (hasLayout)
      return llvm::Error::success();

    IRRewriter rewriter(&getContext());
    for (auto &[planned, ids, commands] : work) {
      rewriter.setInsertionPoint(planned);
      auto lowered = CommandMatMulOp::create(
          rewriter, planned.getLoc(), planned.getResult().getType(),
          planned.getLhs(), planned.getRhs(), planned.getTargetAttr(),
          planned.getTileMAttr(), planned.getTileNAttr(), planned.getTileKAttr(),
          planned.getTiles(), planned.getMemoryPlan(),
          rewriter.getI64IntegerAttr(ids[0]), rewriter.getI64IntegerAttr(ids[1]),
          rewriter.getI64IntegerAttr(ids[2]), encodeCommands(rewriter, commands));
      rewriter.replaceOp(planned, lowered.getResult());
    }
    module->setAttr(dramLayoutAttr, encodeDRAMLayout(rewriter, *layout));
    return llvm::Error::success();
  }
};

}

void mlir::npu::registerDMAInsertionPass() {
  PassRegistration<DMAInsertionPass>();
}
