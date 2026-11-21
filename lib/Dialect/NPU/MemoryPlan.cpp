#include "NPU/MemoryPlan.h"

#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <utility>

using namespace mlir::npu;

namespace {

llvm::Expected<int64_t> alignBytes(int64_t bytes, int64_t alignment) {
  int64_t remainder = bytes % alignment;
  int64_t aligned = bytes;
  if (remainder &&
      llvm::AddOverflow(bytes, alignment - remainder, aligned))
    return llvm::createStringError("aligned byte size overflow");
  return aligned;
}

llvm::Expected<int64_t> dataStart(SRAMConfig config) {
  if (config.capacityBytes <= 0 || config.reservedBytes < 0 ||
      config.reservedBytes >= config.capacityBytes)
    return llvm::createStringError("invalid SRAM capacity or reserved bytes");
  if (config.alignmentBytes <= 0 ||
      !llvm::isPowerOf2_64(config.alignmentBytes))
    return llvm::createStringError("SRAM alignment must be a positive power of two");
  auto start = alignBytes(config.reservedBytes, config.alignmentBytes);
  if (!start)
    return start.takeError();
  if (*start >= config.capacityBytes)
    return llvm::createStringError("aligned reserved region leaves no usable SRAM");
  return *start;
}

struct Requirements {
  std::vector<BufferRequirement> buffers;
  std::vector<TileBufferBinding> bindings;
};

llvm::Expected<Requirements>
deriveRequirements(llvm::ArrayRef<TileDesc> tiles, int64_t alignment) {
  if (tiles.empty() || tiles.size() > static_cast<size_t>(maxTileRecords))
    return llvm::createStringError("invalid tile count for SRAM planning");

  Requirements requirements;
  requirements.buffers.reserve(tiles.size() * 3);
  requirements.bindings.reserve(tiles.size());
  int64_t nextEvent = 0;
  int64_t nextOutput = 0;
  int64_t nextKOffset = 0;
  int64_t accumulator = -1;
  TileDesc firstK{};

  auto addBuffer = [&](BufferRole role, int64_t tileIndex, int64_t outputId,
                       int64_t rows, int64_t columns, int64_t liveBegin,
                       int64_t liveEnd) -> llvm::Expected<int64_t> {
    int64_t elements, rawBytes;
    if (llvm::MulOverflow(rows, columns, elements) ||
        llvm::MulOverflow(elements, int64_t{4}, rawBytes))
      return llvm::createStringError("FP32 buffer byte size overflow");
    auto allocated = alignBytes(rawBytes, alignment);
    if (!allocated)
      return allocated.takeError();
    int64_t id = static_cast<int64_t>(requirements.buffers.size());
    requirements.buffers.push_back({id, role, tileIndex, outputId, rawBytes,
                                    *allocated, alignment, liveBegin, liveEnd});
    return id;
  };

  for (int64_t index = 0, count = static_cast<int64_t>(tiles.size());
       index < count; ++index) {
    const TileDesc &tile = tiles[index];
    if (tile.mSize <= 0 || tile.nSize <= 0 || tile.kSize <= 0 ||
        tile.mOffset < 0 || tile.nOffset < 0 || tile.outputTileId != nextOutput ||
        tile.isFirstK != (accumulator == -1) || tile.kOffset != nextKOffset)
      return llvm::createStringError("invalid tile sequence for SRAM planning");
    if (tile.isFirstK) {
      firstK = tile;
      int64_t zeroC = nextEvent++;
      auto id = addBuffer(BufferRole::C, index, tile.outputTileId, tile.mSize,
                          tile.nSize, zeroC, zeroC);
      if (!id)
        return id.takeError();
      accumulator = *id;
    } else if (tile.mOffset != firstK.mOffset || tile.nOffset != firstK.nOffset ||
               tile.mSize != firstK.mSize || tile.nSize != firstK.nSize) {
      return llvm::createStringError("output tile changes across K segments");
    }

    int64_t loadA = nextEvent++;
    int64_t loadB = nextEvent++;
    int64_t compute = nextEvent++;
    auto inputA = addBuffer(BufferRole::A, index, tile.outputTileId, tile.mSize,
                            tile.kSize, loadA, compute + 1);
    if (!inputA)
      return inputA.takeError();
    auto inputB = addBuffer(BufferRole::B, index, tile.outputTileId, tile.kSize,
                            tile.nSize, loadB, compute + 1);
    if (!inputB)
      return inputB.takeError();
    requirements.bindings.push_back({index, *inputA, *inputB, accumulator});

    if (tile.isLastK) {
      int64_t storeC = nextEvent++;
      requirements.buffers[accumulator].liveEnd = storeC + 1;
      accumulator = -1;
      ++nextOutput;
      nextKOffset = 0;
    } else {
      int64_t followingOffset;
      if (llvm::AddOverflow(nextKOffset, tile.kSize, followingOffset))
        return llvm::createStringError("tile K offset overflow");
      nextKOffset = followingOffset;
    }
  }
  if (accumulator != -1)
    return llvm::createStringError("missing final K segment for SRAM planning");
  return requirements;
}

template <typename GetRequirement>
std::vector<size_t> chronologicalOrder(size_t count,
                                       GetRequirement getRequirement) {
  std::vector<size_t> order(count);
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
    const auto &left = getRequirement(lhs);
    const auto &right = getRequirement(rhs);
    return std::tie(left.liveBegin, left.id) < std::tie(right.liveBegin, right.id);
  });
  return order;
}

using Expiry = std::pair<int64_t, size_t>;
using ExpiryQueue =
    std::priority_queue<Expiry, std::vector<Expiry>, std::greater<Expiry>>;

llvm::Expected<int64_t> computePeak(llvm::ArrayRef<BufferRequirement> requests,
                                    llvm::ArrayRef<size_t> order) {
  ExpiryQueue expiry;
  int64_t peak = 0;
  int64_t liveBytes = 0;
  for (size_t index : order) {
    const auto &request = requests[index];
    while (!expiry.empty() && expiry.top().first <= request.liveBegin) {
      liveBytes -= requests[expiry.top().second].allocatedBytes;
      expiry.pop();
    }
    int64_t newLiveBytes;
    if (llvm::AddOverflow(liveBytes, request.allocatedBytes, newLiveBytes))
      return llvm::createStringError("peak live bytes overflow");
    liveBytes = newLiveBytes;
    peak = std::max(peak, liveBytes);
    expiry.emplace(request.liveEnd, index);
  }
  return peak;
}

void releaseBlock(std::map<int64_t, int64_t> &freeBlocks, int64_t begin,
                  int64_t end) {
  auto next = freeBlocks.lower_bound(begin);
  if (next != freeBlocks.begin()) {
    auto previous = std::prev(next);
    if (previous->second == begin) {
      begin = previous->first;
      freeBlocks.erase(previous);
    }
  }
  if (next != freeBlocks.end() && next->first == end) {
    end = next->second;
    freeBlocks.erase(next);
  }
  freeBlocks.emplace(begin, end);
}

struct Placement {
  std::vector<BufferAllocation> allocations;
  int64_t peak;
  int64_t highWater;
};

llvm::Expected<Placement>
placeBuffers(llvm::ArrayRef<BufferRequirement> requests, SRAMConfig config) {
  auto start = dataStart(config);
  if (!start)
    return start.takeError();
  std::unordered_set<int64_t> ids;
  for (const auto &request : requests) {
    if (request.id < 0 || !ids.insert(request.id).second)
      return llvm::createStringError("invalid or duplicate buffer ID");
    if (request.liveBegin < 0 || request.liveEnd <= request.liveBegin)
      return llvm::createStringError("invalid buffer lifetime");
    if (request.rawBytes <= 0 ||
        request.alignmentBytes != config.alignmentBytes)
      return llvm::createStringError("invalid buffer size or alignment");
    auto allocated = alignBytes(request.rawBytes, config.alignmentBytes);
    if (!allocated)
      return allocated.takeError();
    if (*allocated != request.allocatedBytes)
      return llvm::createStringError("buffer size does not match requirement");
  }
  auto order = chronologicalOrder(requests.size(),
                                  [&](size_t index) -> const BufferRequirement & {
                                    return requests[index];
                                  });
  auto peak = computePeak(requests, order);
  if (!peak)
    return peak.takeError();
  if (*peak > config.capacityBytes - *start)
    return llvm::createStringError(
        llvm::Twine("insufficient SRAM capacity: peak requires ") +
        llvm::Twine(*peak) + " bytes, usable SRAM is " +
        llvm::Twine(config.capacityBytes - *start) + " bytes");

  Placement placement{std::vector<BufferAllocation>(requests.size()), *peak, 0};
  std::map<int64_t, int64_t> freeBlocks{{*start, config.capacityBytes}};
  ExpiryQueue expiry;
  std::unordered_set<int64_t> usedBases;
  for (size_t index : order) {
    const auto &request = requests[index];
    while (!expiry.empty() && expiry.top().first <= request.liveBegin) {
      const auto &old = placement.allocations[expiry.top().second];
      releaseBlock(freeBlocks, old.address,
                   old.address + old.requirement.allocatedBytes);
      expiry.pop();
    }
    auto available = std::find_if(
        freeBlocks.begin(), freeBlocks.end(), [&](const auto &range) {
          return request.allocatedBytes <= range.second - range.first;
        });
    if (available == freeBlocks.end())
      return llvm::createStringError(
          llvm::Twine("First-Fit fragmentation: no contiguous SRAM range for buffer ") +
          llvm::Twine(request.id) + " requiring " +
          llvm::Twine(request.allocatedBytes) + " bytes");
    int64_t address = available->first;
    int64_t end = address + request.allocatedBytes;
    int64_t freeEnd = available->second;
    freeBlocks.erase(available);
    if (end < freeEnd)
      freeBlocks.emplace(end, freeEnd);
    placement.allocations[index] =
        {request, address, !usedBases.insert(address).second};
    placement.highWater = std::max(placement.highWater, end);
    expiry.emplace(request.liveEnd, index);
  }
  return placement;
}

llvm::Error verifyLayout(llvm::ArrayRef<BufferAllocation> allocations,
                         int64_t capacity, int64_t dataBegin,
                         int64_t alignment, int64_t expectedPeak,
                         int64_t expectedHighWater) {
  auto order = chronologicalOrder(
      allocations.size(), [&](size_t index) -> const BufferRequirement & {
        return allocations[index].requirement;
      });
  ExpiryQueue expiry;
  std::map<int64_t, int64_t> activeRanges;
  std::unordered_set<int64_t> usedBases;
  int64_t liveBytes = 0;
  int64_t peak = 0;
  int64_t highWater = 0;
  for (size_t index : order) {
    const auto &allocation = allocations[index];
    const auto &request = allocation.requirement;
    int64_t address = allocation.address;
    if (address < dataBegin || address > capacity ||
        request.allocatedBytes > capacity - address)
      return llvm::createStringError(llvm::Twine("buffer ") +
                                     llvm::Twine(request.id) +
                                     ": buffer address range is outside usable SRAM");
    if (address % alignment)
      return llvm::createStringError(llvm::Twine("buffer ") +
                                     llvm::Twine(request.id) +
                                     ": buffer address is not aligned");
    int64_t end = address + request.allocatedBytes;

    while (!expiry.empty() && expiry.top().first <= request.liveBegin) {
      const auto &old = allocations[expiry.top().second];
      activeRanges.erase(old.address);
      liveBytes -= old.requirement.allocatedBytes;
      expiry.pop();
    }
    auto next = activeRanges.lower_bound(address);
    if ((next != activeRanges.end() && next->first < end) ||
        (next != activeRanges.begin() && std::prev(next)->second > address))
      return llvm::createStringError(llvm::Twine("buffer ") +
                                     llvm::Twine(request.id) +
                                     ": simultaneously live buffers overlap");
    if (allocation.reusedAddress != !usedBases.insert(address).second)
      return llvm::createStringError("incorrect reused_address flag");
    activeRanges.emplace(address, end);
    expiry.emplace(request.liveEnd, index);
    liveBytes += request.allocatedBytes;
    peak = std::max(peak, liveBytes);
    highWater = std::max(highWater, end);
  }
  if (peak != expectedPeak)
    return llvm::createStringError("incorrect peak_live_bytes");
  if (highWater != expectedHighWater)
    return llvm::createStringError("incorrect arena_high_water_bytes");
  return llvm::Error::success();
}

}

llvm::Error mlir::npu::verifySRAMConfig(SRAMConfig config) {
  auto start = dataStart(config);
  return start ? llvm::Error::success() : start.takeError();
}

llvm::Expected<std::vector<BufferAllocation>>
mlir::npu::allocateBuffers(llvm::ArrayRef<BufferRequirement> requirements,
                          SRAMConfig config) {
  auto placement = placeBuffers(requirements, config);
  if (!placement)
    return placement.takeError();
  return std::move(placement->allocations);
}

llvm::Expected<MemoryPlan>
mlir::npu::generateMemoryPlan(llvm::ArrayRef<TileDesc> tiles, SRAMConfig config) {
  if (auto error = verifySRAMConfig(config))
    return error;
  auto requirements = deriveRequirements(tiles, config.alignmentBytes);
  if (!requirements)
    return requirements.takeError();
  auto placement = placeBuffers(requirements->buffers, config);
  if (!placement)
    return placement.takeError();
  return MemoryPlan{config, std::move(placement->allocations),
                    std::move(requirements->bindings), placement->peak,
                    placement->highWater};
}

llvm::Error mlir::npu::verifyMemoryPlan(llvm::ArrayRef<TileDesc> tiles,
                                       const MemoryPlan &plan) {
  auto start = dataStart(plan.config);
  if (!start)
    return start.takeError();
  auto expected = deriveRequirements(tiles, plan.config.alignmentBytes);
  if (!expected)
    return expected.takeError();
  if (plan.allocations.size() != expected->buffers.size())
    return llvm::createStringError("unexpected number of buffer allocations");
  if (plan.bindings.size() != expected->bindings.size())
    return llvm::createStringError("unexpected number of tile bindings");

  std::vector<bool> accounted(expected->buffers.size());
  for (const auto &allocation : plan.allocations) {
    const auto &request = allocation.requirement;
    if (request.id < 0 || static_cast<size_t>(request.id) >= accounted.size())
      return llvm::createStringError("unknown buffer ID");
    if (accounted[request.id])
      return llvm::createStringError("duplicate buffer ID");
    accounted[request.id] = true;
    const auto &required = expected->buffers[request.id];
    if (required.role != request.role || required.tileIndex != request.tileIndex ||
        required.outputTileId != request.outputTileId)
      return llvm::createStringError("buffer role or tile ownership does not match requirement");
    if (required.rawBytes != request.rawBytes ||
        required.allocatedBytes != request.allocatedBytes ||
        required.alignmentBytes != request.alignmentBytes)
      return llvm::createStringError("buffer size does not match requirement");
    if (request.liveBegin < 0 || request.liveEnd <= request.liveBegin)
      return llvm::createStringError("invalid buffer lifetime");
    if (required.liveBegin != request.liveBegin ||
        required.liveEnd != request.liveEnd)
      return llvm::createStringError("buffer lifetime does not match required events");
  }

  auto hasBuffer = [&](int64_t id) {
    return id >= 0 && static_cast<size_t>(id) < accounted.size() && accounted[id];
  };
  for (size_t index = 0; index < plan.bindings.size(); ++index) {
    const auto &binding = plan.bindings[index];
    const auto &required = expected->bindings[index];
    if (!hasBuffer(binding.aBufferId) || !hasBuffer(binding.bBufferId) ||
        !hasBuffer(binding.cBufferId))
      return llvm::createStringError("tile binding references unknown buffer");
    if (binding.tileIndex != required.tileIndex ||
        binding.aBufferId != required.aBufferId ||
        binding.bBufferId != required.bBufferId ||
        binding.cBufferId != required.cBufferId)
      return llvm::createStringError("tile binding does not match required A/B/C buffers");
  }

  return verifyLayout(plan.allocations, plan.config.capacityBytes, *start,
                      plan.config.alignmentBytes, plan.peakLiveBytes,
                      plan.arenaHighWaterBytes);
}
