cmake_minimum_required(VERSION 2.8)
project(Executor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pthread -fPIC")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

include_directories(common)
include_directories(include)

find_package(OpenCL)
find_package(Boost REQUIRED COMPONENTS program_options)

# Enable ExternalProject CMake module
include(ExternalProject)

# Download and install GoogleTest
ExternalProject_Add(
    gtest
    URL https://github.com/google/googletest/archive/release-1.7.0.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    # Disable install step
    INSTALL_COMMAND ""
)

# Create a libgtest target to be used as a dependency by test programs
# add_library(libgtest IMPORTED STATIC GLOBAL)
# add_dependencies(libgtest gtest)

# Set gtest properties
# ExternalProject_Get_Property(gtest source_dir binary_dir)
# set_target_properties(libgtest PROPERTIES
#     "IMPORTED_LOCATION" "${binary_dir}/libgtest.a"
#     "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
#    "INTERFACE_INCLUDE_DIRECTORIES" "${source_dir}/include"
# )
# I couldn't make it work with INTERFACE_INCLUDE_DIRECTORIES
include_directories("${source_dir}/include")

set(SOURCE_FILES
    src/run.cpp
    src/file_utils.cpp
    src/csv_utils.cpp
    src/opencl_utils.cpp
    src/mmio.cpp
    src/sparse_matrix.cpp
    src/kernel.cpp)

add_library (Executor ${SOURCE_FILES})

function(add_app name)
    add_executable(harness_${name} apps/${name}.cpp)
    target_link_libraries(harness_${name} Executor pthread ${OpenCL_LIBRARIES} ${Boost_LIBRARIES})
endfunction()

include_directories(apps)

add_app(mm)
add_app(mv)
add_app(gemv)
add_app(nbody)
add_app(kmeans)
add_app(generic)
add_app(stencil)
add_app(hotspot)
add_app(hotspot3D)
add_app(stencilNoPad)
add_app(convolution)
add_app(convolution2D)
add_app(shocStencil2D)
add_app(convolutionSeparableY)
add_app(srad1)
add_app(srad2)
add_app(acoustic3D)
add_app(spmv)

# enable_testing()
# add_subdirectory(test)
