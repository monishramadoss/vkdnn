add_library(base STATIC  "arithmetic.h" "logical.h" "base.cpp" )

target_link_libraries(base "tensor")
