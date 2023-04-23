// activation.h : Header file for your target.

#pragma once
#include <cmath>
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

struct activation_parameter
{
    float alpha;
    float beta;
    float gamma;
    uint32_t total;
};

inline std::string activation_shader = R"(
layout(push_constant) uniform pushBlock {
    float alpha;
    float beta;
    float gamma;
    uint total;
};

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
)";


inline uint32_t set_group_size(const activation_parameter& p)
{
    return align_size(p.total, 1024) / 1024;
}

inline void hardtanh(const float min_val, const float max_val, const tensor& t1, const tensor& t2)
{
    const activation_parameter p{min_val, max_val, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader,"{1}[i] = clamp({0}[i], alpha, beta);", t1, t2);
    k_runtime->make_job<activation_parameter>("hardtanh", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}


inline void clip(const float min_val, const float max_val, const tensor& t1, const tensor& t2)
{
    const activation_parameter p{ min_val, max_val, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = clamp({0}[i], alpha, beta);", t1, t2);
    k_runtime->make_job<activation_parameter>("clip", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void leaky_relu(float neg_slope, const tensor& t1, const tensor& t2) {
    const activation_parameter p{ neg_slope, 0, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = max({0}[i], 0) + alpha * min(0, {0}[i]);", t1, t2);
    k_runtime->make_job<activation_parameter>("leakyrelu", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void relu(const tensor& t1, const tensor& t2) {
    const activation_parameter p{ 0, 0, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = max({0}[i], 0);", t1, t2);
    k_runtime->make_job<activation_parameter>("relu", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void sigmoid(const tensor& t1, const tensor& t2) {
    const activation_parameter p{ 0, 0, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = 1 / (1. + exp({0}[i]));", t1, t2);
    k_runtime->make_job<activation_parameter>("sigmoid", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}


inline void celu(const float neg_slope, const tensor& t1, const tensor& t2) {
    const activation_parameter p{ neg_slope, 0, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = max({0}[i], 0) + min(0, alpha * (exp({0}[i] / alpha) - 1.));", t1, t2);
    k_runtime->make_job<activation_parameter>("celu", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void elu(const float neg_slope, const tensor& t1, const tensor& t2) {
    const activation_parameter p{ neg_slope, 0, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string shader_code = R"(
if({0}[i] < 0)
    {1}[i] = alpha * (exp({0}[i]) - 1.);
else
    {1}[i] = {0}[i];
)";
    const std::string kernel_code = unary_shader_code(activation_shader, shader_code, t1, t2);
    k_runtime->make_job<activation_parameter>("elu", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}



inline void hardsigmoid(const float alpha, const float beta, const tensor& t1, const tensor& t2)
{
    const activation_parameter p{ alpha, beta, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = max(0., min(1., {0}[i] * alpha + beta));", t1, t2);
    k_runtime->make_job<activation_parameter>("hardsigmoid", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void hardswish(const tensor& t1, const tensor& t2) {
    const activation_parameter p{ 1/6, 0.5, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = {0}[i] * max(0., min(1., {0}[i] * alpha + beta));", t1, t2);
    k_runtime->make_job<activation_parameter>("hardswish", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void selu(const float alpha, const float beta, const float gamma, const tensor& t1, const tensor& t2){
    const activation_parameter p{ alpha, beta, gamma, static_cast<uint32_t>(t1.get_size()) };
    std::string shader_code = R"(
if({0}[i] <= 0)
    {1}[i] = gamma * (alpha * exp({0}[i]) - beta);
else 
    {1}[i] = gamma * {0}[i];
)";
    const std::string kernel_code = unary_shader_code(activation_shader, shader_code, t1, t2);
    k_runtime->make_job<activation_parameter>("selu", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void prelu(const tensor& t1, const tensor& t2, const tensor& t3) {
    const activation_parameter p{ 0, 0, 0, static_cast<uint32_t>(t1.get_size()) };
    std::string shader_code = R"(
if({0}[i] < 0)
    {2}[i] = {1}[i] * {0}[i];
else 
    {2}[i] = {0}[i];
)";
    std::string kernel_code = binary_shader_code(activation_shader, shader_code, t1, t2, t3);
    k_runtime->make_job<activation_parameter>("prelu", kernel_code, { t1.get_data(), t2.get_data(), t3.get_data() },
        p, set_group_size(p));
}


inline void softsigmoid(const tensor& t1, const tensor& t2) {
    const activation_parameter p{ 0, 0, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = log(exp({0}[i]) + 1.f);", t1, t2);
    k_runtime->make_job<activation_parameter>("softsigmoid", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}


inline void softsign(const tensor& t1, const tensor& t2) {
    const activation_parameter p{ 0, 0, 0, static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = {0}[i] / (1 + abs({0}[i]));", t1, t2);
    k_runtime->make_job<activation_parameter>("softsign", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void gelu(const tensor& t1, const tensor& t2) {
    constexpr float PI = 3.1415926535897932384626f;
    const activation_parameter p{ sqrt(2/ PI), 0.044715f, 0, static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(activation_shader, "{1}[i] = 0.5 * {0}[i] * (1 + tanh(alpha * ({0}[i] + beta * {0}[i] * {0}[i] * {0}[i])));", t1, t2);
    k_runtime->make_job<activation_parameter>("gelu", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}
