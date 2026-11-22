add_mlir_library(MLIRNPUCodeGen
  VirtualISA.cpp
  WeightPacker.cpp
  ProgramManifest.cpp

  LINK_LIBS PUBLIC
  MLIRNPUDialect
  MLIRArithDialect
  MLIRIR
)
