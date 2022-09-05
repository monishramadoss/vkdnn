add_library(base STATIC "base.cpp" "binary.h" "logical.h" "translation.h" "reduction.h")
target_compile_features(base PRIVATE cxx_std_20)
target_link_libraries(base "tensor")
