﻿add_library(nn STATIC "nn.cpp" "nn.h" "linear.h" "normalization.h" "reccurent.h")
target_compile_features(nn PRIVATE cxx_std_20)
target_link_libraries(nn "tensor")
