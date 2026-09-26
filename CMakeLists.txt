cmake_minimum_required(VERSION 3.28)
project(RealTimeDecoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(RTD_ENABLE_TESTS "Build unit tests" ON)
option(RTD_ENABLE_ASAN "Build with AddressSanitizer + UBSan" OFF)
option(RTD_ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(RTD_ENABLE_CLANG_TIDY "Run clang-tidy during build" OFF)

add_library(rtd_warnings INTERFACE)
target_compile_options(rtd_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
)

if(RTD_ENABLE_ASAN)
    add_library(rtd_sanitizers INTERFACE)
    target_compile_options(rtd_sanitizers INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(rtd_sanitizers INTERFACE -fsanitize=address,undefined)
elseif(RTD_ENABLE_TSAN)
    add_library(rtd_sanitizers INTERFACE)
    target_compile_options(rtd_sanitizers INTERFACE -fsanitize=thread -fno-omit-frame-pointer -g)
    target_link_options(rtd_sanitizers INTERFACE -fsanitize=thread)
else()
    add_library(rtd_sanitizers INTERFACE)
endif()

if(RTD_ENABLE_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE NAMES clang-tidy REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

add_subdirectory(src)

if(RTD_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
