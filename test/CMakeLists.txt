include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SYSTEM
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(rtd_tests
    unit/arith_test.cpp
    unit/buffer_test.cpp
    unit/graph_test.cpp
    unit/priors_test.cpp
    unit/observables_test.cpp
    unit/config_test.cpp
    unit/gamma_test.cpp
    unit/kernels_test.cpp
    unit/executor_test.cpp
    unit/relay_test.cpp
    unit/relay_limits_test.cpp
    unit/convergence_rows_test.cpp
    unit/sink_test.cpp
    unit/properties_test.cpp
    unit/npy_test.cpp
    unit/loaders_test.cpp
    unit/window_time_test.cpp
    unit/window_plan_test.cpp
    unit/window_property_test.cpp
    unit/window_stream_test.cpp
    unit/window_toy_test.cpp
    unit/allocation_counter.cpp
    golden/golden_test.cpp
    golden/window_golden_test.cpp
)
target_include_directories(rtd_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rtd_tests PRIVATE rtd_io rtd_window rtd_warnings GTest::gtest_main)
target_compile_definitions(rtd_tests PRIVATE
    RTD_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
# Fixed-point arithmetic: policy, decodes against a naive integer decoder, emulator goldens.
target_sources(rtd_tests PRIVATE
    unit/fixed_arith_test.cpp
    golden/fixed_golden_test.cpp)
if(RTD_ENABLE_HARNESS)
    target_sources(rtd_tests PRIVATE unit/harness_fixed_test.cpp)
endif()
# Selection policies and confidence values: rules, controller hooks, window actions.
target_sources(rtd_tests PRIVATE
    unit/selection_test.cpp)
if(RTD_ENABLE_HARNESS)
    target_sources(rtd_tests PRIVATE
        unit/harness_test.cpp
        unit/harness_spec_test.cpp
        unit/harness_recording_test.cpp
        unit/harness_cli_test.cpp
        unit/harness_sliding_test.cpp
        unit/harness_selection_test.cpp)
    target_link_libraries(rtd_tests PRIVATE rtd_harness)
    # The in-memory decoding API, against rtd_decode on the same shots.
    target_sources(rtd_tests PRIVATE unit/api_test.cpp)
    target_link_libraries(rtd_tests PRIVATE rtd_api)
endif()
rtd_enable_tidy(rtd_tests)

include(GoogleTest)
gtest_discover_tests(rtd_tests DISCOVERY_TIMEOUT 120)
