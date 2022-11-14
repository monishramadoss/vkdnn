add_library(tensor STATIC "tensor.cpp" "../TESTING/view.h"  "tensor.h" "init.h" "copy.h" "utils.h"  "view.h")
target_compile_features(tensor PRIVATE cxx_std_20)
target_link_libraries(tensor "runtime")
