// logical.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

struct logic_parameter
{
    uint32_t total;
};

template<class T>
inline void eq(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("eq", "{2}[i] = {0}[i] == val;", t1, t2, t3);
}
template<class T>
inline void ne(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("ne", "{2}[i] = {0}[i] != val;", t1, t2, t3);
}
template<class T>
inline void lt(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("lt", "{2}[i] = {0}[i] < val;", t1, t2, t3);
}
template<class T>
inline void gt(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("gt", "{2}[i] = {0}[i] > val;", t1, t2, t3);
}
template<class T>
inline void le(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("le", "{2}[i] = {0}[i] <= val;", t1, t2, t3);
}
template<class T>
inline void ge(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("ge", "{2}[i] = {0}[i] >= val;", t1, t2, t3);
}

inline std::string logic_shader = R"(
layout(push_constant) uniform pushBlock {
    uint total;
};

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
)";

inline uint32_t set_group_size(const logic_parameter& p)
{
    return align_size(p.total, 1024) / 1024;
}

template<>
inline void eq(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t3.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");


    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] == {1}[i];", t1, t2, t3);
    k_runtime->make_job<logic_parameter>("eq", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                     p, set_group_size(p));
}

template<>
inline void ne(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t3.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] != {1}[i];", t1, t2, t3);
    k_runtime->make_job<logic_parameter>("ne", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                     p, set_group_size(p));
}

template<>
inline void lt(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t3.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] < {1}[i];", t1, t2, t3);
    k_runtime->make_job<logic_parameter>("lt", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                     p, set_group_size(p));
}

template<>
inline void gt(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t3.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] > {1}[i];", t1, t2, t3);
    k_runtime->make_job<logic_parameter>("gt", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                     p, set_group_size(p));
}

template<>
inline void le(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t3.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] <= {1}[i];", t1, t2, t3);
    k_runtime->make_job<logic_parameter>("le", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                     p, set_group_size(p));
}

template<>
inline void ge(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t3.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] >= {1}[i];", t1, t2, t3);
    k_runtime->make_job<logic_parameter>("ge", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                     p, set_group_size(p));
}

inline void bang(const tensor& t1, const tensor& t2)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t2.get_type() != BOOL || t1.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = unary_shader_code(binary_shader, "{1}[i] = !{0}[i];", t1, t2);
    k_runtime->make_job<logic_parameter>("bang", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void logical_xor(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t3.get_type() != BOOL || t1.get_type() != BOOL || t2.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] ^^ {1}[i];", t1, t2, t3);
    k_runtime->make_job<logic_parameter>("xor", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                     p, set_group_size(p));
}

inline void logical_and(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t3.get_type() != BOOL || t1.get_type() != BOOL || t2.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] && {1}[i];", t1, t2, t3);
    k_runtime->make_job<logic_parameter>("and", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                     p, set_group_size(p));
}

inline void logical_or(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const logic_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (t3.get_type() != BOOL || t1.get_type() != BOOL || t2.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {1}[i] || {1}[i];", t1, t2, t3);
    k_runtime->make_job<logic_parameter>("or", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                     p, set_group_size(p));
}

