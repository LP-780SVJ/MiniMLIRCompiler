#ifndef NPU_VIRTUALISA_H
#define NPU_VIRTUALISA_H

#include "NPU/SchedulePlanAttrs.h"
#include "llvm/Support/JSON.h"

namespace mlir::npu {

inline constexpr int64_t programSchemaVersion = 1;
inline constexpr int64_t maxJSONInteger = (int64_t{1} << 53) - 1;
inline constexpr TileShape virtualISANativeTile{32, 32, 32};

llvm::Expected<llvm::json::Object>
encodeVirtualProgram(const ScheduleInput &input, const ExecutionSchedule &schedule);

}

#endif
