add_mlir_library(MLIRNPUTiling
  NPUTiling.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRNPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)

add_mlir_library(MLIRNPUSRAMPlanner
  SRAMPlanner.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRNPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)

add_mlir_library(MLIRNPUDMAInsertion
  DMAInsertion.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRNPUDialect
  MLIRIR
  MLIRPass
)

add_mlir_library(MLIRNPUScheduler
  NPUScheduler.cpp
  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  MLIRNPUDialect
  MLIRIR
  MLIRPass
)
