add_mlir_library(MLIRMiniTransforms
  AddReluFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Mini

  LINK_LIBS PUBLIC
  MLIRMini
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)
