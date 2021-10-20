cmake_minimum_required(VERSION 3.1.0)
project(interop LANGUAGES CXX CUDA)

find_package(OpenCV 3.2 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Vulkan REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})


# Warnings / Errors
add_compile_options(-std=c++17 -Wall -Wno-unused-function -Wno-unused-variable
  -Wno-unused-but-set-variable -Wno-unused-but-set-variable -Wno-write-strings
  -Wno-deprecated -Wno-deprecated-declarations -Wno-unknown-pragmas)

# Don't follow symlinks when FILE GLOB_RECURSE (and don't warn)
cmake_policy(SET CMP0009 NEW)

set(CUDA_HOST_COMPILER "/usr/bin/g++")
set(BUILD_TESTS OFF CACHE BOOL "Build unit tests")

set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")
include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-D_FORCE_INLINES;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-g")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

CUDA_COMPILE(cuda cudafuns.cu)


add_executable(main
               main.cpp
               VulkanBase.cpp
               ${cuda}
               )

target_include_directories(main PUBLIC ${Vulkan_INCLUDE_DIRS})
# target_compile_features(main PRIVATE cxx_std_17)
target_include_directories(main PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(main
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    Vulkan::Vulkan
    -lglfw
    -lcuda)