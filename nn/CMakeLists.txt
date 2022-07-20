add_library(nn STATIC "nn.cpp" "nn.h")

target_link_libraries(nn "tensor")
