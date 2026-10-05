cmake_minimum_required(VERSION 3.28)
project(rtd VERSION 0.1.0 DESCRIPTION "Real-time Relay-BP decoder for qLDPC codes" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# No C++ modules here; skipping the dependency scan speeds up builds and keeps
# compile_commands.json readable by clang tooling.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(RTD_ENABLE_TESTS "Build unit and golden tests" ON)
option(RTD_ENABLE_BENCH "Build the google-benchmark suite" ON)
option(RTD_ENABLE_HARNESS "Build the rtd_decode batch driver" ON)
option(RTD_ENABLE_CUDA "Build the CUDA backend (not implemented yet)" OFF)
option(RTD_NATIVE "Compile for the host CPU (-march=native)" ON)
option(RTD_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(RTD_ENABLE_ASAN "Build with AddressSanitizer + UBSan" OFF)
option(RTD_ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(RTD_ENABLE_CLANG_TIDY "Run clang-tidy during build" OFF)
option(RTD_ENABLE_PYTHON "Build the Python extension module rtd._native (needs nanobind)" OFF)

if(RTD_ENABLE_ASAN AND RTD_ENABLE_TSAN)
    message(FATAL_ERROR "RTD_ENABLE_ASAN and RTD_ENABLE_TSAN are mutually exclusive")
endif()
if(RTD_ENABLE_PYTHON)
    if(NOT RTD_ENABLE_HARNESS)
        message(FATAL_ERROR "RTD_ENABLE_PYTHON needs RTD_ENABLE_HARNESS (the spec parser)")
    endif()
    # The static libraries are linked into a shared extension module.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

include(RtdOptions)

if(RTD_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    set(RTD_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

add_subdirectory(src)

if(RTD_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
