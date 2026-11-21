#ifndef NPU_VIRTUALNPUPROFILE_H
#define NPU_VIRTUALNPUPROFILE_H

#include "NPU/TilePlan.h"
#include "llvm/Support/JSON.h"

namespace mlir::npu {

llvm::Expected<llvm::json::Object> loadTargetProfile(llvm::StringRef path);
llvm::Expected<TileShape> readNativeLimits(const llvm::json::Object &profile);

}

#endif
