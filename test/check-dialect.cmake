execute_process(COMMAND "${DRIVER}" --show-dialects
  RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT status EQUAL 0 OR NOT "${output}${error}" MATCHES "Available Dialects: [^\n]*mini")
  message(FATAL_ERROR "mini dialect was not listed: ${output}${error}")
endif()
