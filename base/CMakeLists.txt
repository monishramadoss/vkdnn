add_library(base STATIC "base.cpp" "binary.h" "logical.h" "transform.h" "reduction.h" "padding.h")
target_compile_features(base PRIVATE cxx_std_20)
target_link_libraries(base "tensor")
