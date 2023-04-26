add_library(tensor STATIC "tensor.cpp" "tensor.h" "init.h" "utils.h" "view.h" )
target_compile_features(tensor PRIVATE cxx_std_20)
target_link_libraries(tensor "runtime")
