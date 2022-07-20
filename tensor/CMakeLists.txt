add_library(tensor STATIC "tensor.cpp" "tensor.h")


target_link_libraries(tensor "runtime")
