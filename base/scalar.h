// scalar.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

template<typename T>
struct scalar_binary_parameter {
    T val;
    uint32_t total;
};

inline std::string unary_shader_2 = R"(
layout(push_constant) uniform pushBlock {
    {0} val;
    uint total;
};

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
)";

template<typename T>
int set_group_size(scalar_binary_parameter<T> p) {
    return align_size(p.total, 1024) / 1024;
}

template<typename T>
inline void scalar_binary_kernel_helper(std::string fn_name, std::string fn_code, std::string param_type, const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_parameter<T> p{ t2, static_cast<uint32_t>(t1.get_size()) };
    std::string _code = Format(unary_shader_2, param_type);
    const std::string kernel_code = unary_shader_code(unary_shader_2, fn_code, t1, t3);
    k_runtime->make_job<scalar_binary_parameter<T>>(fn_name, kernel_code, { t1.get_data(), t3.get_data() }, p,
        set_group_size<T>(p));
}

template<typename T>
inline void scalar_binary_kernel(const tensor& t1, const T& t2, const tensor& t3) {
};


inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const int8_t& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::INT8, t2_typename);
    scalar_binary_kernel_helper<int8_t>(fn_name, fn_code, t2_typename, t1, t2, t3);
}


inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const uint8_t& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::UINT8, t2_typename);
    scalar_binary_kernel_helper<uint8_t>(fn_name, fn_code, t2_typename, t1, t2, t3);
}


inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const int16_t& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::HINT, t2_typename);
    scalar_binary_kernel_helper<int16_t>(fn_name, fn_code, t2_typename, t1, t2, t3);
}


inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const uint16_t& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::HUINT, t2_typename);
    scalar_binary_kernel_helper<uint16_t>(fn_name, fn_code, t2_typename, t1, t2, t3);
}

inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const int32_t& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::INT, t2_typename); 
    scalar_binary_kernel_helper<int32_t>(fn_name, fn_code, t2_typename, t1, t2, t3);
}


inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const  uint32_t& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::UINT, t2_typename);
    scalar_binary_kernel_helper<uint32_t>(fn_name, fn_code, t2_typename, t1, t2, t3);
}


inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const int64_t& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::LINT, t2_typename);
    scalar_binary_kernel_helper<int64_t>(fn_name, fn_code, t2_typename, t1, t2, t3);
}



inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const  uint64_t& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::LUINT, t2_typename);
    scalar_binary_kernel_helper<uint64_t>(fn_name, fn_code, t2_typename, t1, t2, t3);
}


template<typename T>
inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const bool& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::BOOL, t2_typename);
    scalar_binary_kernel_helper<uint32_t>(fn_name, fn_code, t2_typename, t1, t2, t3);
}

template<typename T>
inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const float& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::FLOAT, t2_typename);
    scalar_binary_kernel_helper<float>(fn_name, fn_code, t2_typename, t1, t2, t3);
}

template<typename T>
inline void scalar_binary_kernel(std::string fn_name, std::string fn_code, const tensor& t1, const double& t2, const tensor& t3) {
    std::string t2_typename;
    gen_type(DTYPE::DOUBLE, t2_typename);
    scalar_binary_kernel_helper<double>(fn_name, fn_code, t2_typename, t1, t2, t3);
}




