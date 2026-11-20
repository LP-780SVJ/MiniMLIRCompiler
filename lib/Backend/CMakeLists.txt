add_mlir_library(MLIRNPUTiling
  NPUTiling.cpp

  LINK_LIBS PUBLIC
  MLIRNPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
