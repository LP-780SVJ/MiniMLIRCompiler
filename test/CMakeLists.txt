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
add_test(NAME mini-verifier-regression
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/verifier/check-verifier.sh
    $<TARGET_FILE:mini-opt>)
add_test(NAME mini-verifier-diagnostics
  COMMAND $<TARGET_FILE:mini-opt> --verify-diagnostics
    ${CMAKE_CURRENT_SOURCE_DIR}/verifier/diagnostics.mlir -o /dev/null)
add_test(NAME mini-fusion-regression
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/transforms/check-fusion.sh
    $<TARGET_FILE:mini-opt>)
add_test(NAME mini-lowering-regression
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/conversion/check-lowering.sh
    $<TARGET_FILE:mini-opt>)
add_test(NAME mini-full-pipeline-regression
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/conversion/check-full-pipeline.sh
    $<TARGET_FILE:mini-opt>)

add_test(NAME mini-matmul-valid
  COMMAND $<TARGET_FILE:mini-opt> ${CMAKE_CURRENT_SOURCE_DIR}/matmul/valid.mlir -o /dev/null)
add_test(NAME mini-matmul-diagnostics
  COMMAND $<TARGET_FILE:mini-opt> --split-input-file --verify-diagnostics
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/diagnostics.mlir -o /dev/null)

set(MINI_ONNX_PYTHON "" CACHE FILEPATH "Python interpreter with ONNX for importer tests")
add_test(NAME npu-matmul-valid
  COMMAND $<TARGET_FILE:mini-opt> ${CMAKE_CURRENT_SOURCE_DIR}/npu/valid.mlir -o /dev/null)
add_test(NAME npu-matmul-diagnostics
  COMMAND $<TARGET_FILE:mini-opt> --split-input-file --verify-diagnostics
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/diagnostics.mlir -o /dev/null)
add_test(NAME mini-to-npu-conversion
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/npu/conversion/check-mini-to-npu.sh
    $<TARGET_FILE:mini-opt>)
add_test(NAME npu-tile-ir-valid
  COMMAND $<TARGET_FILE:mini-opt> ${CMAKE_CURRENT_SOURCE_DIR}/npu/tiling/valid.mlir -o /dev/null)
add_test(NAME npu-tile-ir-diagnostics
  COMMAND $<TARGET_FILE:mini-opt> --split-input-file --verify-diagnostics
    ${CMAKE_CURRENT_SOURCE_DIR}/npu/tiling/diagnostics.mlir -o /dev/null)

add_executable(mini-sram-plan-test npu/sram/MemoryPlanTest.cpp)
llvm_update_compile_flags(mini-sram-plan-test)
target_link_libraries(mini-sram-plan-test PRIVATE MLIRNPUDialect)
add_test(NAME npu-sram-allocation COMMAND $<TARGET_FILE:mini-sram-plan-test>)

add_executable(mini-command-plan-test npu/dma/CommandPlanTest.cpp)
llvm_update_compile_flags(mini-command-plan-test)
target_link_libraries(mini-command-plan-test PRIVATE MLIRNPUDialect)
add_test(NAME npu-dma-address-arithmetic COMMAND $<TARGET_FILE:mini-command-plan-test>)

if(MINI_ONNX_PYTHON)
  foreach(case IN ITEMS positive negative onnx)
    add_test(NAME npu-dma-${case}
      COMMAND "${MINI_ONNX_PYTHON}" ${CMAKE_CURRENT_SOURCE_DIR}/npu/dma/check_dma_insertion.py
        $<TARGET_FILE:mini-opt> ${case})
  endforeach()
  foreach(case IN ITEMS positive negative onnx)
    add_test(NAME npu-sram-${case}
      COMMAND "${MINI_ONNX_PYTHON}" ${CMAKE_CURRENT_SOURCE_DIR}/npu/sram/check_sram_plan.py
        $<TARGET_FILE:mini-opt> ${case})
  endforeach()
  foreach(case IN ITEMS positive negative onnx)
    add_test(NAME npu-tiling-${case}
      COMMAND "${MINI_ONNX_PYTHON}" ${CMAKE_CURRENT_SOURCE_DIR}/npu/tiling/check_npu_tiling.py
        $<TARGET_FILE:mini-opt> ${case})
  endforeach()
  foreach(case IN ITEMS matmul-import two-matmul-import weighted-import reject-unsupported)
    add_test(NAME onnx-${case}
      COMMAND "${MINI_ONNX_PYTHON}" ${CMAKE_CURRENT_SOURCE_DIR}/e2e/check_matmul_import.py
        $<TARGET_FILE:mini-opt> ${case})
  endforeach()
  add_test(NAME onnx-golden-artifacts
    COMMAND "${MINI_ONNX_PYTHON}" ${CMAKE_CURRENT_SOURCE_DIR}/e2e/check_reference.py artifacts)
  add_test(NAME onnx-output-comparator
    COMMAND "${MINI_ONNX_PYTHON}" ${CMAKE_CURRENT_SOURCE_DIR}/e2e/check_reference.py comparator)
  foreach(case IN ITEMS positive negative)
    add_test(NAME onnx-to-npu-${case}
      COMMAND "${MINI_ONNX_PYTHON}" ${CMAKE_CURRENT_SOURCE_DIR}/e2e/check_onnx_to_npu_ir.py
        $<TARGET_FILE:mini-opt> ${case})
  endforeach()
endif()
