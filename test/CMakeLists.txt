include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(rtd_tests
    decoder_test.cpp
)

target_link_libraries(rtd_tests PRIVATE
    realtimedecoder
    rtd_warnings
    rtd_sanitizers
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(rtd_tests)
