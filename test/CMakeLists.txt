add_test(NAME mini-dialect-registered
  COMMAND ${CMAKE_COMMAND}
    -DDRIVER=$<TARGET_FILE:mini-opt>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check-dialect.cmake)
add_test(NAME mini-parse-valid
  COMMAND $<TARGET_FILE:mini-opt> ${CMAKE_CURRENT_SOURCE_DIR}/valid.mlir)
add_test(NAME mini-parse-ping
  COMMAND $<TARGET_FILE:mini-opt> ${CMAKE_CURRENT_SOURCE_DIR}/mini-ping.mlir)
add_test(NAME mini-reject-unknown-op
  COMMAND $<TARGET_FILE:mini-opt> --verify-diagnostics
    ${CMAKE_CURRENT_SOURCE_DIR}/unknown-mini-op.mlir)
add_test(NAME mini-reject-i32-tensor
  COMMAND $<TARGET_FILE:mini-opt> --verify-diagnostics
    ${CMAKE_CURRENT_SOURCE_DIR}/dialect/invalid-type.mlir)
add_test(NAME mini-reject-dynamic-shape
  COMMAND $<TARGET_FILE:mini-opt> --verify-diagnostics
    ${CMAKE_CURRENT_SOURCE_DIR}/dialect/invalid-dynamic.mlir)
add_test(NAME mini-reject-extra-operand
  COMMAND $<TARGET_FILE:mini-opt> --verify-diagnostics
    ${CMAKE_CURRENT_SOURCE_DIR}/dialect/invalid-arity.mlir)
