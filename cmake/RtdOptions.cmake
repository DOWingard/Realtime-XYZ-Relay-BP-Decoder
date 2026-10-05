# Interface targets carrying the project-wide compile settings.

add_library(rtd_warnings INTERFACE)
target_compile_options(rtd_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wnon-virtual-dtor
    -Wold-style-cast -Wcast-align -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
    -Wformat=2 -Wimplicit-fallthrough -Wmisleading-indentation
)
if(RTD_WARNINGS_AS_ERRORS)
    target_compile_options(rtd_warnings INTERFACE -Werror)
endif()

# Arithmetic settings every translation unit that instantiates a decoder kernel must share.
# Without -ffp-contract=off the compiler may fuse a*b + c into one FMA, which rounds once instead
# of twice and breaks bit-identity with the goldens; fast-math would also let it reassociate sums.
add_library(rtd_arith_flags INTERFACE)
target_compile_options(rtd_arith_flags INTERFACE -ffp-contract=off -fno-fast-math)
if(RTD_NATIVE)
    target_compile_options(rtd_arith_flags INTERFACE -march=native)
endif()

add_library(rtd_sanitizers INTERFACE)
if(RTD_ENABLE_ASAN)
    target_compile_options(rtd_sanitizers INTERFACE
        -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
    target_link_options(rtd_sanitizers INTERFACE -fsanitize=address,undefined)
elseif(RTD_ENABLE_TSAN)
    target_compile_options(rtd_sanitizers INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
    target_link_options(rtd_sanitizers INTERFACE -fsanitize=thread)
endif()

# Applies clang-tidy to one target when RTD_ENABLE_CLANG_TIDY is on. Third-party targets are
# never passed here.
function(rtd_enable_tidy target)
    if(RTD_CLANG_TIDY)
        set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "${RTD_CLANG_TIDY}")
    endif()
endfunction()
