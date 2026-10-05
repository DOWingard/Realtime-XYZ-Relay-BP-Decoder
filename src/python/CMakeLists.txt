# rtd._native: the nanobind extension module over rtd_api.
#
# The interpreter defaults to the project's virtual environment (python/.venv), whose nanobind
# provides the CMake package; pass -DPython_EXECUTABLE=... for another one. With
# -DRTD_PYTHON_OUTPUT_DIR=<repo>/python/src/rtd the module is written straight into the package,
# where `import rtd` finds it.

set(RTD_PYTHON_OUTPUT_DIR "" CACHE PATH "Directory to write the rtd._native module to")

if(NOT DEFINED Python_EXECUTABLE AND EXISTS "${PROJECT_SOURCE_DIR}/python/.venv/bin/python")
    set(Python_EXECUTABLE "${PROJECT_SOURCE_DIR}/python/.venv/bin/python")
endif()
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE rtd_nanobind_status ERROR_QUIET)
if(NOT rtd_nanobind_status EQUAL 0)
    message(FATAL_ERROR
        "nanobind is not installed for ${Python_EXECUTABLE}; install the optional dependency "
        "group 'bindings' of python/pyproject.toml (nanobind) into that environment")
endif()
find_package(nanobind CONFIG REQUIRED)

# NB_SUPPRESS_WARNINGS includes nanobind's headers as system headers, so the project's warning
# set applies to this module's own code only; NOMINSIZE keeps the project's optimisation level.
nanobind_add_module(rtd_python NB_STATIC NB_SUPPRESS_WARNINGS NOMINSIZE module.cpp)
set_target_properties(rtd_python PROPERTIES OUTPUT_NAME _native)
if(RTD_PYTHON_OUTPUT_DIR)
    set_target_properties(rtd_python PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${RTD_PYTHON_OUTPUT_DIR}")
endif()
target_link_libraries(rtd_python PRIVATE rtd_api rtd_warnings)
rtd_enable_tidy(rtd_python)
