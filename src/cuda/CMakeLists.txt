enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# These policies are related to SWIG python libraries naming
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# SWIG requires the Python header
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
# TODO: Is NumPy required?
find_package(NumPy REQUIRED)
include_directories(${NumPy_INCLUDE_DIRS})

# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(radonusfft.i PROPERTIES CPLUSPLUS ON)
swig_add_library(radonusfft LANGUAGE python SOURCES radonusfft.i radonusfft.cu)

set_source_files_properties(ptychofft.i PROPERTIES CPLUSPLUS ON)
swig_add_library(ptychofft LANGUAGE python SOURCES ptychofft.i ptychofft.cu)

set_source_files_properties(deform.i PROPERTIES CPLUSPLUS ON)
swig_add_library(deform LANGUAGE python SOURCES deform.i deform.cu)

swig_link_libraries(radonusfft cudart cufft)
swig_link_libraries(ptychofft cudart cufft)
swig_link_libraries(deform cudart nppig nppial nppc npps nppidei)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/radonusfft.py" DESTINATION ptychotomo)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/ptychofft.py" DESTINATION ptychotomo)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/deform.py" DESTINATION ptychotomo)

install(TARGETS radonusfft ptychofft deform LIBRARY DESTINATION ptychotomo)