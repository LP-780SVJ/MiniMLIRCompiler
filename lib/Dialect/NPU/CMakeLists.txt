add_mlir_dialect_library(MLIRNPUDialect
  NPUDialect.cpp
  NPUOps.cpp
  TilePlan.cpp
  TilePlanAttrs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NPU

  DEPENDS
  MLIRNPUOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
