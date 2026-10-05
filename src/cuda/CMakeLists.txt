# Device backend. Only the interface exists so far, and it compiles as plain C++ so that the
# seam is checked on machines without nvcc; the device code will add the CUDA language here.
add_library(rtd_cuda STATIC backend_cuda.cpp)
add_library(rtd::cuda ALIAS rtd_cuda)
target_link_libraries(rtd_cuda PUBLIC rtd_core PRIVATE rtd_warnings)
rtd_enable_tidy(rtd_cuda)
