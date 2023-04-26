add_library(base STATIC "base.cpp" "base.h" "binary.h" "cc.h" "logical.h" "reduction.h" "scalar.h" "transform.h" "unary.h")
target_compile_features(base PRIVATE cxx_std_20)
target_link_libraries(base "tensor")
