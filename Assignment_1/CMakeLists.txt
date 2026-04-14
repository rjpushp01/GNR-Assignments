cmake_minimum_required(VERSION 3.10)
project(my_backend)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Pybind11
find_package(pybind11 REQUIRED)

# Find OpenCV
set(OpenCV_DIR "/home/pushp-raj/anaconda3/envs/gnr638_ass/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# Include directories
include_directories(include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
file(GLOB SOURCES "src/*.cpp")

# Create the extension module
pybind11_add_module(my_backend ${SOURCES})

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(my_backend PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Link OpenCV
target_link_libraries(my_backend PRIVATE ${OpenCV_LIBS})

# Output the library to the python package directory
set_target_properties(my_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python")

