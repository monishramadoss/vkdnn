# CMakeList.txt : CMake project for vulkan_dnn, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.12)

project ("vulkan_dnn")

# Add source to this project's executable.
add_executable (vulkan_dnn "prototype.cpp" "vulkan_dnn.h")

target_link_libraries(vulkan_dnn  "nn" "tensor")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET vulkan_dnn PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
add_subdirectory("runtime")


add_subdirectory("tensor")

add_subdirectory("nn")

add_subdirectory("blas")

add_subdirectory("base")
