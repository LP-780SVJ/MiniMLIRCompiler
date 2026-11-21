#include "NPU/VirtualNPUProfile.h"
#include "NPU/TilePlanAttrs.h"

#include "llvm/Support/MemoryBuffer.h"
#include <limits>
#include <optional>
#include <utility>

using namespace mlir::npu;
using llvm::StringRef;

llvm::Expected<llvm::json::Object> mlir::npu::loadTargetProfile(StringRef path) {
  if (path.empty())
    return llvm::createStringError("requires a config path");
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError("cannot read target config: %s",
                                   buffer.getError().message().c_str());
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return parsed.takeError();
  auto *profile = parsed->getAsObject();
  const auto *schema = profile ? profile->get("schema_version") : nullptr;
  if (!schema || schema->getAsUINT64() != 1)
    return llvm::createStringError("expected target schema_version 1");
  if (profile->getString("target") != virtualNPUTarget)
    return llvm::createStringError("expected target virtual-npu-v1");
  return std::move(*profile);
}

llvm::Expected<TileShape>
mlir::npu::readNativeLimits(const llvm::json::Object &profile) {
  const auto *compute = profile.getObject("compute");
  if (!compute)
    return llvm::createStringError("missing compute object");
  const auto *units = compute->get("matrix_units");
  if (!units || units->getAsUINT64() != 1)
    return llvm::createStringError("expected one matrix unit");
  if (compute->getString("input_dtype") != "float32" ||
      compute->getString("accumulator_dtype") != "float32")
    return llvm::createStringError("expected float32 input and accumulator");
  if (compute->getString("matmul_semantics") != "accumulate")
    return llvm::createStringError("expected accumulate MatMul semantics");

  const auto *native = compute->getObject("native_matmul_tile");
  if (!native)
    return llvm::createStringError("missing compute.native_matmul_tile object");
  TileShape limits{};
  for (auto [name, limit] : {std::pair{"m", &limits.m}, std::pair{"n", &limits.n},
                            std::pair{"k", &limits.k}}) {
    const auto *value = native->get(name);
    auto size = value ? value->getAsUINT64() : std::nullopt;
    if (!size || *size == 0 ||
        *size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return llvm::createStringError(
          "native_matmul_tile.%s must be a positive i64 integer", name);
    *limit = static_cast<int64_t>(*size);
  }
  return limits;
}

llvm::Expected<SRAMConfig>
mlir::npu::readSRAMConfig(const llvm::json::Object &profile) {
  const auto *memory = profile.getObject("memory");
  if (!memory)
    return llvm::createStringError("missing memory object");
  if (memory->getString("layout") != "row_major")
    return llvm::createStringError("expected row_major SRAM layout");
  const auto *dma = profile.getObject("dma");
  if (!dma || dma->getString("mode") != "synchronous" ||
      dma->getString("buffering") != "single")
    return llvm::createStringError("expected synchronous DMA with single buffering");

  auto readBytes = [memory](StringRef name) -> std::optional<int64_t> {
    const auto *field = memory->get(name);
    auto value = field ? field->getAsUINT64() : std::nullopt;
    if (!value || *value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    return static_cast<int64_t>(*value);
  };
  auto capacity = readBytes("sram_bytes");
  auto reserved = readBytes("reserved_bytes");
  auto alignment = readBytes("alignment_bytes");
  if (!capacity || !reserved || !alignment)
    return llvm::createStringError(
        "memory.sram_bytes, reserved_bytes and alignment_bytes must be nonnegative i64 integers");

  SRAMConfig config{*capacity, *reserved, *alignment};
  if (auto error = verifySRAMConfig(config))
    return error;
  return config;
}
