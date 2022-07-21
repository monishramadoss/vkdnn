add_library(base STATIC  "arithmetic.h" "logical.h" "base.cpp" )
target_compile_features(base PRIVATE cxx_std_20)

target_link_libraries(base "tensor")
