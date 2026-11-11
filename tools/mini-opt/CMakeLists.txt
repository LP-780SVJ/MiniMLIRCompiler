
set(LIBS
  MLIRMini
  MLIRMiniTransforms
  MLIRMiniToLinalg
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIROptLib
  MLIRTransforms
)

add_llvm_executable(mini-opt
  mini-opt.cpp
)

llvm_update_compile_flags(mini-opt)

target_link_libraries(mini-opt
  PRIVATE
  ${LIBS}
)

mlir_check_all_link_libraries(mini-opt)
