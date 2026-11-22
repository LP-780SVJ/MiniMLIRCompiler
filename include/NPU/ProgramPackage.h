#ifndef NPU_PROGRAMPACKAGE_H
#define NPU_PROGRAMPACKAGE_H

#include "NPU/VirtualISA.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::npu {

struct PackedWeights {
  std::string bytes;
  llvm::json::Array entries;
};

struct ProgramPackage {
  std::string program, weights, manifest;
};

/// Pack constants from a verified single-entry module and its matching layout.
llvm::Expected<PackedWeights> packWeights(ModuleOp module, const DRAMLayout &layout);
llvm::Expected<ProgramPackage>
buildProgramPackage(ModuleOp module, llvm::StringRef targetProfile,
                    llvm::StringRef modelMetadataPath);

}

#endif
