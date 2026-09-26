add_library(realtimedecoder
    decoder.cpp
)

target_include_directories(realtimedecoder PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(realtimedecoder PRIVATE rtd_warnings rtd_sanitizers)
target_link_libraries(realtimedecoder PUBLIC rtd_sanitizers)
