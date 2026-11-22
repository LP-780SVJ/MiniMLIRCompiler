#include "NPU/VirtualISA.h"
#include "NPU/TilePlanAttrs.h"

using namespace mlir::npu;
using llvm::json::Array;
using llvm::json::Object;

namespace {

Object encodePayload(const ZeroCommand &cmd) {
  return Object{{"opcode", "ZERO"}, {"buffer_id", cmd.bufferId},
          {"dst_sram", cmd.dstSram}, {"rows", cmd.rows}, {"cols", cmd.cols}};
}

Object encodePayload(const DMALoad2DCommand &cmd) {
  return Object{{"opcode", "DMA_LOAD_2D"}, {"tensor_id", cmd.tensorId},
          {"buffer_id", cmd.bufferId}, {"src_dram", cmd.srcDram},
          {"dst_sram", cmd.dstSram}, {"rows", cmd.rows}, {"row_bytes", cmd.rowBytes},
          {"src_stride_bytes", cmd.srcStrideBytes},
          {"dst_stride_bytes", cmd.dstStrideBytes}};
}

Object encodePayload(const MatMulTileCommand &cmd) {
  return Object{{"opcode", "MATMUL_TILE"}, {"a_buffer_id", cmd.aBufferId},
          {"b_buffer_id", cmd.bBufferId}, {"c_buffer_id", cmd.cBufferId},
          {"a_sram", cmd.aSram}, {"b_sram", cmd.bSram}, {"c_sram", cmd.cSram},
          {"m", cmd.m}, {"n", cmd.n}, {"k", cmd.k}, {"accumulate", cmd.accumulate}};
}

Object encodePayload(const DMAStore2DCommand &cmd) {
  return Object{{"opcode", "DMA_STORE_2D"}, {"tensor_id", cmd.tensorId},
          {"buffer_id", cmd.bufferId}, {"src_sram", cmd.srcSram},
          {"dst_dram", cmd.dstDram}, {"rows", cmd.rows}, {"row_bytes", cmd.rowBytes},
          {"src_stride_bytes", cmd.srcStrideBytes},
          {"dst_stride_bytes", cmd.dstStrideBytes}};
}

}

llvm::Expected<Object>
mlir::npu::encodeVirtualProgram(const ScheduleInput &input,
                                const ExecutionSchedule &schedule) {
  if (auto error = verifySchedule(input.layout, input.matmuls, schedule))
    return error;
  std::vector<const ScheduledCommand *> ordered(schedule.commands.size());
  for (const auto &command : schedule.commands)
    ordered[command.sequenceIndex] = &command;

  Array instructions;
  instructions.reserve(ordered.size());
  for (const auto *command : ordered) {
    auto [matmulIndex, localIndex] = command->origin;
    const auto &local = input.matmuls[matmulIndex].commands[localIndex];
    auto instruction = std::visit(
        [](const auto &payload) { return encodePayload(payload); }, local.payload);
    instruction["command_id"] = command->commandId;
    instruction["sequence_index"] = command->sequenceIndex;
    instruction["matmul_index"] = matmulIndex;
    instruction["local_event_index"] = localIndex;
    instruction["tile_index"] = local.tileIndex;
    instruction["dependencies"] = Array(command->dependencies);
    instructions.push_back(std::move(instruction));
  }

  return Object{{"schema_version", programSchemaVersion},
                {"target", virtualNPUTarget},
                {"schedule_contract", serialSRAMSchedule},
                {"instruction_count", static_cast<int64_t>(instructions.size())},
                {"instructions", std::move(instructions)}};
}
