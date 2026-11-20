add_mlir_conversion_library(MLIRMiniToLinalg
  MiniToLinalg.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Mini

  LINK_LIBS PUBLIC
  MLIRMini
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

add_mlir_conversion_library(MLIRMiniToNPU
  MiniToNPU.cpp
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Mini

  LINK_LIBS PUBLIC
  MLIRMini
  MLIRNPUDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
