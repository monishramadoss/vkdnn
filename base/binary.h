// binary.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"
#include "scalar.h"
struct binary_parameter
{
    uint32_t total;
};

inline std::string binary_shader = R"(
layout(push_constant) uniform pushBlock {
    uint total;
};

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in; 
)";

inline uint32_t set_group_size(const binary_parameter& p)
{
    return align_size(p.total, 1024) / 1024;
}

template<class T>
inline void add(const tensor& t1, const T& t2, const tensor&t3) {
    scalar_binary_kernel<T>("add", "{2}[i] = {0}[i] + val;", t1, t2, t3);
}
template<class T>
inline void sub(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("sub", "{2}[i] = {0}[i] - val;", t1, t2, t3);
}
template<class T>
inline void mul(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("mul", "{2}[i] = {0}[i] * val;", t1, t2, t3);
}
template<class T>
inline void div(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("div", "{2}[i] = {0}[i] / val;", t1, t2, t3);
}
template<class T>
inline void pow(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("pow", "{2}[i] = pow({0}[i], val);", t1, t2, t3);
}
template<class T>
inline void max(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("max", "{2}[i] = max({0}[i], val);", t1, t2, t3);
}
template<class T>
inline void min(const tensor& t1, const T& t2, const tensor& t3) {
    scalar_binary_kernel<T>("min", "{2}[i] = min({0}[i], val);", t1, t2, t3);
}


template <>
inline void add(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] + {1}[i];", t1, t2, t3);
    k_runtime->make_job<binary_parameter>("add", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                      p, set_group_size(p));
}

template <>
inline void sub(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] - {1}[i];", t1, t2, t3);
    k_runtime->make_job<binary_parameter>("sub", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                      p, set_group_size(p));
}

template <>
inline void mul(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] * {1}[i];", t1, t2, t3);
    k_runtime->make_job<binary_parameter>("mul", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                      p, set_group_size(p));
}

template <>
inline void div(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] / {1}[i];", t1, t2, t3);
    k_runtime->make_job<binary_parameter>("div", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                      p, set_group_size(p));
}

inline void remainder(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = mod({0}[i], {1}[i]);", t1, t2, t3);
    k_runtime->make_job<binary_parameter>("mod", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                      p, set_group_size(p));
}

inline void fmod(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] - trunc({0}[i] / {1}[i]) * {1}[i];", t1, t2, t3);
    k_runtime->make_job<binary_parameter>("fmod", kernel_code, { t1.get_data(), t2.get_data(), t3.get_data() },
        p, set_group_size(p));
}

template <>
inline void pow(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = pow({0}[i], {1}[i]);", t1, t2, t3);
    k_runtime->make_job<binary_parameter>("pow", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                      p, set_group_size(p));
}

template <>
inline void max(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = max({0}[i], {1}[i]);", t1, t2, t3);
    k_runtime->make_job<binary_parameter>("max", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                      p, set_group_size(p));
}

template <>
inline void min(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = min({0}[i], {1}[i]);", t1, t2, t3);
    k_runtime->make_job<binary_parameter>("min", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
                                                      p, set_group_size(p));
}


inline void atan2(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const binary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    std::string shader_code = R"(
const float PI = 3.1415926535897932384626433832795;

if({0}[i] > 0)
    {2}[i] = atan({1}[i]/{0}[i]);
if({0}[i] < 0 && {1}[i] >= 0 )
    {2}[i] = atan({1}[i]/{0}[i]) + PI;
if({0}[i] < 0 && {1}[i] < 0)
    {2}[i] = atan({1}[i]/{0}[i]) - PI;
if({0}[i] == 0 && {1}[i] > 0)
    {2}[i] = PI / 2.f;
if({0}[i] == 0 && {1}[i] < 0)
    {2}[i] = -PI / 2.f;
if({0}[i] == 0 && {1}[i] == 0)
    {2}[i] = 1.f/0.f;
)";

    const std::string kernel_code = binary_shader_code(binary_shader, shader_code, t1, t2, t3);
    k_runtime->make_job<binary_parameter>("atan2", kernel_code, { t1.get_data(), t2.get_data(), t3.get_data() },
        p, set_group_size(p));
}
