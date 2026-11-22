#include "NPU/ProgramPackage.h"

#include "NPU/NPUOps.h"
#include "NPU/TilePlanAttrs.h"
#include "NPU/VirtualNPUProfile.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

using namespace mlir;
using namespace mlir::npu;
using llvm::json::Array;
using llvm::json::Object;

namespace {

StringRef storageKind(TensorStorageKind kind) {
  switch (kind) {
  case TensorStorageKind::Input: return "input";
  case TensorStorageKind::Constant: return "constant";
  case TensorStorageKind::Intermediate: return "intermediate";
  case TensorStorageKind::Output: return "output";
  }
  llvm_unreachable("unknown tensor storage kind");
}

std::string sha256(StringRef bytes) {
  llvm::SHA256 hash;
  hash.update(bytes);
  return llvm::toHex(hash.final(), /*LowerCase=*/true);
}

std::string formatJSON(Object object) {
  return llvm::formatv("{0:2}\n", llvm::json::Value(std::move(object))).str();
}

llvm::Expected<Object> readMetadata(StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(), "cannot read model metadata");
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return parsed.takeError();
  auto *metadata = parsed->getAsObject();
  if (!metadata || metadata->getInteger("schema_version") != 1 ||
      metadata->getString("target") != virtualNPUTarget ||
      metadata->getString("dtype") != "float32")
    return llvm::createStringError("incompatible model metadata schema, target or dtype");
  return std::move(*metadata);
}

llvm::Expected<Array> encodePorts(const DRAMLayout &layout, ArrayRef<int64_t> ids,
                                  StringRef prefix, const Object *metadata) {
  const Array *names = nullptr;
  const Object *shapes = nullptr;
  if (metadata) {
    names = metadata->getArray((prefix + "_names").str());
    shapes = metadata->getObject((prefix + "s").str());
    if (!names || !shapes || names->size() != ids.size() ||
        shapes->size() != ids.size())
      return llvm::createStringError(llvm::Twine("metadata ") + prefix +
                                     " counts do not match MLIR ABI");
  }

  Array ports;
  llvm::SmallDenseSet<StringRef> seen;
  for (auto [index, id] : llvm::enumerate(ids)) {
    std::string name;
    if (metadata) {
      auto text = (*names)[index].getAsString();
      if (!text || text->empty() || text->contains('\0') || !seen.insert(*text).second)
        return llvm::createStringError(llvm::Twine("invalid or duplicate metadata ") +
                                       prefix + " name");
      const Array *shape = shapes->getArray(*text);
      const auto &tensor = layout.tensors[id];
      if (!shape || shape->size() != 2 || (*shape)[0].getAsInteger() != tensor.rows ||
          (*shape)[1].getAsInteger() != tensor.cols)
        return llvm::createStringError(llvm::Twine("metadata ") + prefix +
                                       " shape/order mismatch for " + *text);
      name = text->str();
    } else {
      name = (prefix + "_" + llvm::Twine(index)).str();
    }
    ports.push_back(Object{{"name", std::move(name)}, {"tensor_id", id}});
  }
  return ports;
}

bool sameTile(TileShape lhs, TileShape rhs) {
  return lhs.m == rhs.m && lhs.n == rhs.n && lhs.k == rhs.k;
}

llvm::Error verifyTarget(ModuleOp module, const ScheduleInput &input,
                         const Object &profile, TileShape native, SRAMConfig sram) {
  if (!sameTile(native, virtualISANativeTile))
    return llvm::createStringError("Virtual ISA v1 requires native tile 32x32x32");
  if (profile.getObject("memory")->getString("dram_model") != "linear_bytes" ||
      profile.getObject("dma")->getBoolean("supports_2d_strided") != true)
    return llvm::createStringError("expected linear byte DRAM and 2D strided DMA");
  if (sram.capacityBytes > maxJSONInteger || input.layout.totalBytes > maxJSONInteger)
    return llvm::createStringError("memory capacity exceeds the exact JSON integer range");

  size_t index = 0;
  for (Operation &op : module.getBody()->front().getRegion(0).front()) {
    auto matmul = dyn_cast<CommandMatMulOp>(op);
    if (!matmul)
      continue;
    TileShape tile{matmul.getTileMAttr().getInt(), matmul.getTileNAttr().getInt(),
                   matmul.getTileKAttr().getInt()};
    const auto &actual = input.matmuls[index++].memory.config;
    if (matmul.getTargetAttr().getValue() != virtualNPUTarget ||
        !sameTile(tile, native) || actual.capacityBytes != sram.capacityBytes ||
        actual.reservedBytes != sram.reservedBytes ||
        actual.alignmentBytes != sram.alignmentBytes)
      return llvm::createStringError("scheduled command plan does not match target profile");
  }
  return llvm::Error::success();
}

}

llvm::Expected<ProgramPackage>
mlir::npu::buildProgramPackage(ModuleOp module, StringRef targetProfile,
                              StringRef modelMetadataPath) {
  if (!module->hasAttr(executionScheduleAttr))
    return llvm::createStringError("missing npu.execution_schedule; run npu-schedule first");
  auto input = readScheduleInput(module);
  if (!input)
    return input.takeError();
  auto schedule = decodeSchedule(module->getAttrOfType<DictionaryAttr>(executionScheduleAttr));
  if (!schedule)
    return schedule.takeError();

  auto profile = parseTargetProfile(targetProfile);
  if (!profile)
    return profile.takeError();
  auto native = readNativeLimits(*profile);
  if (!native)
    return native.takeError();
  auto sram = readSRAMConfig(*profile);
  if (!sram)
    return sram.takeError();
  if (auto error = verifyTarget(module, *input, *profile, *native, *sram))
    return error;
  auto program = encodeVirtualProgram(*input, *schedule);
  if (!program)
    return program.takeError();

  std::optional<Object> metadata;
  if (!modelMetadataPath.empty()) {
    auto read = readMetadata(modelMetadataPath);
    if (!read)
      return read.takeError();
    metadata.emplace(std::move(*read));
  }
  auto &layout = input->layout;
  const auto *names = metadata ? &*metadata : nullptr;
  auto inputs = encodePorts(layout, layout.inputIds, "input", names);
  if (!inputs)
    return inputs.takeError();
  auto outputs = encodePorts(layout, layout.outputIds, "output", names);
  if (!outputs)
    return outputs.takeError();
  auto weights = packWeights(module, layout);
  if (!weights)
    return weights.takeError();

  Array tensors;
  for (const auto &tensor : layout.tensors) {
    tensors.push_back(Object{{"id", tensor.id},
                             {"kind", storageKind(tensor.kind)},
                             {"base", tensor.base},
                             {"shape", Array{tensor.rows, tensor.cols}},
                             {"row_stride_bytes", tensor.rowStrideBytes},
                             {"size_bytes", tensor.sizeBytes}});
  }
  auto programBytes = formatJSON(std::move(*program));
  Object manifest{
      {"schema_version", 1},
      {"program_schema_version", programSchemaVersion},
      {"target", virtualNPUTarget},
      {"target_profile_sha256", sha256(targetProfile)},
      {"entry", "main"},
      {"dtype", "float32"},
      {"byte_order", "little"},
      {"layout", "row_major"},
      {"schedule_contract", serialSRAMSchedule},
      {"hardware", Object{{"native_tile", Array{native->m, native->n, native->k}},
                          {"sram_bytes", sram->capacityBytes},
                          {"reserved_bytes", sram->reservedBytes},
                          {"sram_alignment_bytes", sram->alignmentBytes},
                          {"dram_bytes", layout.totalBytes},
                          {"dram_alignment_bytes", layout.alignmentBytes}}},
      {"tensors", std::move(tensors)},
      {"inputs", std::move(*inputs)},
      {"outputs", std::move(*outputs)},
      {"weights", Object{{"file", "weights.bin"},
                         {"size_bytes", static_cast<uint64_t>(weights->bytes.size())},
                         {"sha256", sha256(weights->bytes)},
                         {"entries", std::move(weights->entries)}}},
      {"program", Object{{"file", "program.json"},
                         {"instruction_count", static_cast<uint64_t>(schedule->commands.size())},
                         {"size_bytes", static_cast<uint64_t>(programBytes.size())},
                         {"sha256", sha256(programBytes)}}}};
  return ProgramPackage{std::move(programBytes), std::move(weights->bytes),
                        formatJSON(std::move(manifest))};
}
