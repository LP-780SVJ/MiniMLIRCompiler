add_mlir_conversion_library(MLIRMiniToLinalg
  MiniToLinalg.cpp

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
