FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
    URL_HASH SHA256=b334658edd35efcf06a99d9be21e4e93e092bd5f95074c1673d5c8705d95c104
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SYSTEM
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)
# Clang 22 reports the library's __COUNTER__ probe as a C2y extension, which its own
# -pedantic-errors turns into an error.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-Wno-c2y-extensions RTD_HAVE_WNO_C2Y_EXTENSIONS)
if(RTD_HAVE_WNO_C2Y_EXTENSIONS)
    target_compile_options(benchmark PRIVATE -Wno-c2y-extensions)
    target_compile_options(benchmark_main PRIVATE -Wno-c2y-extensions)
endif()

add_executable(rtd_bench bench.cpp)
target_link_libraries(rtd_bench PRIVATE rtd_io benchmark::benchmark rtd_warnings)
if(RTD_ENABLE_HARNESS)
    target_link_libraries(rtd_bench PRIVATE rtd_harness)
    target_compile_definitions(rtd_bench PRIVATE RTD_BENCH_CAN_PIN=1)
endif()
rtd_enable_tidy(rtd_bench)

# Fixed-point message formats against f32: seconds per iteration on the calling thread (unpinned).
add_executable(rtd_bench_fixed fixed_bench.cpp)
target_link_libraries(rtd_bench_fixed PRIVATE rtd_io benchmark::benchmark rtd_warnings)
rtd_enable_tidy(rtd_bench_fixed)
