add_library(blas STATIC "blas.cpp" "blas.h")
target_link_libraries(blas "runtime" "tensor")
target_compile_features(blas PRIVATE cxx_std_20)
