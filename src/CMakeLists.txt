include(FetchContent)

# ---- rtd_core: the decoder. Standard library only. ------------------------------------------
add_library(rtd_core STATIC
    core/graph.cpp
    core/priors.cpp
    core/observables.cpp
    core/config.cpp
    core/gamma.cpp
    core/executor.cpp
    core/instantiations.cpp
    core/instantiations_fixed.cpp
    core/selection.cpp
)
add_library(rtd::core ALIAS rtd_core)
target_include_directories(rtd_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(rtd_core PUBLIC cxx_std_23)
target_link_libraries(rtd_core
    PUBLIC rtd_arith_flags rtd_sanitizers
    PRIVATE rtd_warnings)
find_package(Threads REQUIRED)
target_link_libraries(rtd_core PUBLIC Threads::Threads)
rtd_enable_tidy(rtd_core)

# ---- rtd_window: time structure, window plans, stream decoder. Standard library only. -------
# Merged priors are folded in a fixed order, so this library shares the kernels' no-contraction
# arithmetic flags.
add_library(rtd_window STATIC
    window/error.cpp
    window/spec.cpp
    window/problem.cpp
    window/time_structure.cpp
    window/plan.cpp
    window/stream.cpp
    window/history.cpp
)
add_library(rtd::window ALIAS rtd_window)
target_link_libraries(rtd_window
    PUBLIC rtd_core rtd_arith_flags
    PRIVATE rtd_warnings)
rtd_enable_tidy(rtd_window)

# ---- rtd_io: artifact, shots and golden loaders. ---------------------------------------------
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    SYSTEM
)
FetchContent_MakeAvailable(nlohmann_json)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

add_library(rtd_io STATIC
    io/error.cpp
    io/npy.cpp
    io/sha256.cpp
    io/manifest.cpp
    io/artifact.cpp
    io/shots.cpp
    io/golden.cpp
)
add_library(rtd::io ALIAS rtd_io)
target_link_libraries(rtd_io
    PUBLIC rtd_core nlohmann_json::nlohmann_json
    PRIVATE rtd_warnings OpenSSL::Crypto)
rtd_enable_tidy(rtd_io)

# ---- rtd_cuda: device backend, reserved. ----------------------------------------------------
if(RTD_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()

# ---- rtd_harness: batch driver building blocks and the rtd_decode executable. ---------------
if(RTD_ENABLE_HARNESS)
    add_library(rtd_harness STATIC
        harness/logger.cpp
        harness/spec.cpp
        harness/batch.cpp
        harness/writer.cpp
        harness/system.cpp
        harness/recording.cpp
        harness/sliding.cpp
        harness/selection.cpp
        harness/cli.cpp
    )
    add_library(rtd::harness ALIAS rtd_harness)
    target_include_directories(rtd_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/harness/include)
    # std::stacktrace lives in a separate library in libstdc++ (stdc++exp), whichever compiler
    # uses it; probe instead of keying on the compiler.
    include(CheckCXXSourceCompiles)
    set(rtd_stacktrace_probe
        "#include <stacktrace>\nint main() { return std::stacktrace::current().empty() ? 1 : 0; }")
    set(CMAKE_REQUIRED_FLAGS -std=c++23)
    check_cxx_source_compiles("${rtd_stacktrace_probe}" RTD_STACKTRACE_LINKS)
    set(rtd_stacktrace_library "")
    if(NOT RTD_STACKTRACE_LINKS)
        set(CMAKE_REQUIRED_LIBRARIES stdc++exp)
        check_cxx_source_compiles("${rtd_stacktrace_probe}" RTD_STACKTRACE_LINKS_WITH_STDCXXEXP)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(NOT RTD_STACKTRACE_LINKS_WITH_STDCXXEXP)
            message(FATAL_ERROR "std::stacktrace does not link, with or without stdc++exp")
        endif()
        set(rtd_stacktrace_library stdc++exp)
    endif()
    unset(CMAKE_REQUIRED_FLAGS)
    target_link_libraries(rtd_harness
        PUBLIC rtd_io rtd_window ${rtd_stacktrace_library} PRIVATE rtd_warnings)

    # Provenance for the run record.
    find_package(Git QUIET)
    set(RTD_GIT_REVISION "unknown")
    if(GIT_FOUND)
        execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty --abbrev=12
                        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                        OUTPUT_VARIABLE RTD_GIT_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
    endif()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" rtd_build_type_upper)
    get_target_property(rtd_arith_options rtd_arith_flags INTERFACE_COMPILE_OPTIONS)
    string(JOIN " " rtd_kernel_flags ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${rtd_build_type_upper}}
           ${rtd_arith_options})
    set_source_files_properties(harness/system.cpp PROPERTIES COMPILE_DEFINITIONS
        "RTD_GIT_REVISION=\"${RTD_GIT_REVISION}\";RTD_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\";RTD_KERNEL_FLAGS=\"${rtd_kernel_flags}\"")
    if(RTD_ENABLE_CUDA)
        target_compile_definitions(rtd_harness PUBLIC RTD_HAVE_CUDA_BACKEND=1)
        target_link_libraries(rtd_harness PUBLIC rtd_cuda)
    endif()
    rtd_enable_tidy(rtd_harness)

    add_executable(rtd_decode harness/main.cpp)
    target_link_libraries(rtd_decode PRIVATE rtd_harness rtd_warnings)
    rtd_enable_tidy(rtd_decode)

    # rtd_window_plan: builds a window plan for an artifact and dumps it (the harness supplies
    # only the JSON-lines logger).
    add_executable(rtd_window_plan tools/window_plan_main.cpp)
    target_link_libraries(rtd_window_plan PRIVATE rtd_io rtd_window rtd_harness rtd_warnings)
    rtd_enable_tidy(rtd_window_plan)

    # rtd_select_replay: replays a run's recorded solutions through the selection rules and
    # writes every confidence value per decode.
    add_executable(rtd_select_replay tools/select_replay_main.cpp)
    target_link_libraries(rtd_select_replay PRIVATE rtd_io rtd_window rtd_harness rtd_warnings)
    rtd_enable_tidy(rtd_select_replay)

    # rtd_api: the decoders of rtd_decode over problems and shots held in memory, with no files
    # and no logging (errors are values); the Python bindings wrap it.
    add_library(rtd_api STATIC
        api/error.cpp
        api/problem.cpp
        api/batch.cpp
        api/sources.cpp
        api/decoder.cpp
        api/windowed.cpp
    )
    add_library(rtd::api ALIAS rtd_api)
    target_link_libraries(rtd_api PUBLIC rtd_harness rtd_window rtd_io PRIVATE rtd_warnings)
    rtd_enable_tidy(rtd_api)

    if(RTD_ENABLE_PYTHON)
        add_subdirectory(python)
    endif()
endif()

# ---- rtd_bench ------------------------------------------------------------------------------
if(RTD_ENABLE_BENCH)
    add_subdirectory(bench)
endif()
