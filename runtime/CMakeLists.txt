find_package(Vulkan REQUIRED)
add_library(runtime STATIC  "runtime.h" "device.h"   "allocator.h" "job.h" "runtime.cpp" )
target_include_directories(runtime PUBLIC $<BUILD_INTERFACE:${Vulkan_INCLUDE_DIR}>)
target_link_libraries(runtime ${Vulkan_LIBRARY})
