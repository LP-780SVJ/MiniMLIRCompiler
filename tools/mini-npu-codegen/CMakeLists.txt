add_llvm_executable(mini-npu-codegen mini-npu-codegen.cpp)
llvm_update_compile_flags(mini-npu-codegen)
target_link_libraries(mini-npu-codegen PRIVATE
  MLIRNPUCodeGen
  MLIRMini
  MLIRFuncDialect
  MLIRParser
)
mlir_check_all_link_libraries(mini-npu-codegen)

add_custom_command(
  TARGET mini-npu-codegen POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    "${PROJECT_SOURCE_DIR}/python/runtime/validate_package.py"
    "$<TARGET_FILE_DIR:mini-npu-codegen>/mini-npu-validate-package.py"
  VERBATIM
)
set_property(TARGET mini-npu-codegen APPEND PROPERTY LINK_DEPENDS
  "${PROJECT_SOURCE_DIR}/python/runtime/validate_package.py")
