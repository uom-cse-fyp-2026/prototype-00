cmake_minimum_required(VERSION 3.20)
project(NeuroMindsPrototype VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(NM_ENABLE_MLIR "Compile the actual MLIR executor objects" ON)
option(NM_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
set(NM_CONFIG "${CMAKE_SOURCE_DIR}/configs/prototype.json" CACHE FILEPATH "Approved configuration")
set(NM_LLVM_BIN "" CACHE PATH "Directory containing matching MLIR/LLVM tools")
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE NM_GIT_REV OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT NM_GIT_REV)
  set(NM_GIT_REV uncommitted)
endif()
file(GLOB_RECURSE NM_FINGERPRINT_FILES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cpp" "${CMAKE_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_SOURCE_DIR}/compiler/*.cpp" "${CMAKE_SOURCE_DIR}/tools/*.py"
  "${CMAKE_SOURCE_DIR}/CMakeLists.txt" "${CMAKE_SOURCE_DIR}/compiler/CMakeLists.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${NM_FINGERPRINT_FILES})
set(NM_HASH_INPUT "${CMAKE_CXX_FLAGS}|${CMAKE_CXX_FLAGS_RELEASE}|${NM_SANITIZERS}")
foreach(NM_SOURCE_FILE IN LISTS NM_FINGERPRINT_FILES)
  file(SHA256 "${NM_SOURCE_FILE}" NM_FILE_HASH)
  string(APPEND NM_HASH_INPUT "${NM_FILE_HASH}")
endforeach()
string(SHA256 NM_SOURCE_HASH "${NM_HASH_INPUT}")
set(GEN "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GEN}")
if(NM_ENABLE_MLIR)
  find_program(NM_MLIR_OPT mlir-opt HINTS "${NM_LLVM_BIN}" REQUIRED)
  find_program(NM_MLIR_TRANSLATE mlir-translate HINTS "${NM_LLVM_BIN}" REQUIRED)
  find_program(NM_LLC llc HINTS "${NM_LLVM_BIN}" REQUIRED)
  find_program(NM_LLVM_OPT opt HINTS "${NM_LLVM_BIN}" REQUIRED)
  add_custom_command(OUTPUT "${GEN}/frontend.json" "${GEN}/linear.torch.mlir" "${GEN}/linear.linalg.mlir"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/export_linear.py" --output "${GEN}"
    DEPENDS tools/export_linear.py VERBATIM)
  set(FRONTEND_ARGS --frontend "${GEN}/frontend.json")
  set(FRONTEND_DEPS "${GEN}/frontend.json")
else()
  set(FRONTEND_ARGS --scalar-only)
endif()
add_custom_command(OUTPUT "${GEN}/nm_config.hpp" "${GEN}/executors.mlir" "${GEN}/build_manifest.json"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/generate.py"
    --config "${NM_CONFIG}" --output "${GEN}" ${FRONTEND_ARGS}
  DEPENDS "${NM_CONFIG}" tools/generate.py tools/mlir_kernels.py ${FRONTEND_DEPS} VERBATIM)
add_custom_target(nm_generated DEPENDS "${GEN}/nm_config.hpp" "${GEN}/executors.mlir" "${GEN}/build_manifest.json")
if(NM_ENABLE_MLIR)
  find_package(MLIR REQUIRED CONFIG HINTS "${NM_LLVM_BIN}/../lib/cmake/mlir")
  add_subdirectory(compiler)
  add_custom_command(OUTPUT "${GEN}/executors.o" "${GEN}/executors.llvm.mlir" "${GEN}/executors.ll"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/lower.py"
      --input "${GEN}/executors.mlir" --output "${GEN}" --mlir-opt "${NM_MLIR_OPT}"
      --nm-opt "$<TARGET_FILE:nm-opt>" --mlir-translate "${NM_MLIR_TRANSLATE}" --llc "${NM_LLC}" --llvm-opt "${NM_LLVM_OPT}"
    DEPENDS "${GEN}/executors.mlir" tools/lower.py nm-opt "${NM_MLIR_OPT}" "${NM_LLC}" "${NM_LLVM_OPT}" VERBATIM)
  set_source_files_properties("${GEN}/executors.o" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
  set(MLIR_OBJECT "${GEN}/executors.o")
endif()
add_library(neurominds
  src/quantization/quantization.cpp src/kernels/kernels.cpp src/inspector/inspector.cpp
  src/runtime/policy.cpp src/runtime/cache.cpp src/executor/executor.cpp
  src/benchmark/experiments.cpp src/runtime/io.cpp ${MLIR_OBJECT})
add_dependencies(neurominds nm_generated)
target_include_directories(neurominds PUBLIC include "${GEN}")
target_compile_definitions(neurominds PUBLIC NM_HAS_MLIR=$<BOOL:${NM_ENABLE_MLIR}>
  NM_GIT_REV="${NM_GIT_REV}" NM_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  NM_SOURCE_HASH="${NM_SOURCE_HASH}"
  NM_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(neurominds PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
  if(NM_SANITIZERS)
    target_compile_options(neurominds PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(neurominds PUBLIC -fsanitize=address,undefined)
  endif()
endif()
add_executable(neurominds_demo src/main.cpp)
target_link_libraries(neurominds_demo PRIVATE neurominds)
include(CTest)
if(BUILD_TESTING)
  add_executable(neurominds_tests tests/test_core.cpp)
  target_link_libraries(neurominds_tests PRIVATE neurominds)
  add_test(NAME numerical_and_runtime COMMAND neurominds_tests)
  add_test(NAME generator_validation COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_generator.py")
  add_test(NAME cli_rejects_invalid_shape COMMAND neurominds_demo run --variant fp32 --input 0)
  set_tests_properties(cli_rejects_invalid_shape PROPERTIES WILL_FAIL TRUE)
  if(NM_ENABLE_MLIR)
    add_test(NAME torch_mlir_frontend COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/export_linear.py"
      --output "${CMAKE_BINARY_DIR}/frontend-test" --check)
    add_test(NAME compiler_contract COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_compiler.py"
      --nm-opt "$<TARGET_FILE:nm-opt>" --generated "${GEN}")
  endif()
endif()
install(TARGETS neurominds_demo RUNTIME DESTINATION bin)
