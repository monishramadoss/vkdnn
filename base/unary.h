// unary.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

struct unary_parameter
{
    uint32_t total;
};

inline std::string unary_shader = R"(
layout(push_constant) uniform pushBlock {
    uint total;
};

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
)";

inline uint32_t set_group_size(const unary_parameter& p)
{
    return align_size(p.total, 1024) / 1024;
}

inline void sin(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = sin({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("sin", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void asin(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = asin({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("asin", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void cos(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = cos({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("cos", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void acos(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = acos({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("acos", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void tan(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = tan({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("tan", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void atan(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = atan({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("atan", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void sinh(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = sinh({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("sinh", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void asinh(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = asinh({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("asinh", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void cosh(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = cosh({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("cosh", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void acosh(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = acosh({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("acosh", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void tanh(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = tanh({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("tanh", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void atanh(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = atanh({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("atanh", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}


inline void neg(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = -{0}[i];", t1, t2);
    k_runtime->make_job<unary_parameter>("neg", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void inv(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = 1/{0}[i];", t1, t2);
    k_runtime->make_job<unary_parameter>("inv", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void log(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = log({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("log", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void log2(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = log2({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("log2", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void logp1(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = log({0}[i] + 1.f);", t1, t2);
    k_runtime->make_job<unary_parameter>("logp1", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void mish(const tensor& t1, const tensor& t2) {
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = {0}[i] * tanh( log(exp({0}[i]) + 1.f) )", t1, t2);
    k_runtime->make_job<unary_parameter>("mish", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void log10(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = log({0}[i]) * 0.43429448190325176;", t1, t2);
    k_runtime->make_job<unary_parameter>("log10", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void exp(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = exp({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("exp", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void expm1(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = exp({0}[i]) - 1;", t1, t2);
    k_runtime->make_job<unary_parameter>("expm1", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void exp2(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = exp2({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("exp2", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void sqrt(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = sqrt({1}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("sqrt", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void rsqrt(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = inversesqrt({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("inv_sqrt", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}


inline void increment(const tensor& t1)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = singlton_shader_code(unary_shader, "++{0}[i];", t1);
    k_runtime->make_job<unary_parameter>("increment", kernel_code, {t1.get_data(), }, p,
                                                     set_group_size(p));
}

inline void decrement(const tensor& t1)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    const std::string kernel_code = singlton_shader_code(unary_shader, "--{0}[i];", t1);
    k_runtime->make_job<unary_parameter>("decrement", kernel_code, {t1.get_data(), }, p,
                                                     set_group_size(p));
}

inline void ceil(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (const auto t2_type = t2.get_type(); t2_type == FLOAT || t2_type == DOUBLE || t2_type == BOOL || t2_type ==
        HFLOAT || t2_type == NONE)
        throw std::runtime_error("Return time is not correct");

    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = ceil({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("ceil", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}

inline void floor(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
    if (const auto t2_type = t2.get_type(); t2_type == FLOAT || t2_type == DOUBLE || t2_type == BOOL || t2_type ==
        HFLOAT || t2_type == NONE)
        throw std::runtime_error("Return time is not correct");

    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = floor({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("floor", kernel_code, {t1.get_data(), t2.get_data()}, p,
                                                     set_group_size(p));
}


inline void sign(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = sign({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("sign", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}


inline void signbit(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = sign({0}[i]) != 0;", t1, t2);
    k_runtime->make_job<unary_parameter>("sign", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}


inline void trunc(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = trunc({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("trunc", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void round(const tensor& t1, const tensor& t2)
{
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = round({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("round", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void reShape(const std::vector<uint32_t> new_shape, const tensor& t1, tensor& t2)
{
    t2.reshape(new_shape);
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = {0}[i];", t1, t2);
    k_runtime->make_job<unary_parameter>("reshape", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}
inline void resize(const std::vector<uint32_t> new_shape, const tensor& t1, tensor& t2) {
    t2.reshape(new_shape);
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = {0}[i];", t1, t2);
    k_runtime->make_job<unary_parameter>("reshape", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void squeeze(const std::vector<uint32_t>& axis, const tensor& t1, tensor& t2)
{
    std::vector<uint32_t> new_shape;
    uint32_t i = 0;
    uint32_t j = 0;
    if (axis.size() == 0) {
        for (i = 0; i < t1.get_dims(); ++j, ++i) {
            if (t1.get_shape(i) != 1)
                new_shape.push_back(t1.get_shape(i));
        }
    }
    while (axis.size() != 0) {
        if (i >= axis.size() || j >= t1.get_dims())
            break;
        uint32_t a = axis[i];
        uint32_t s = t1.get_shape(j);
        if (a == j) {
            if (s != 1) {
                new_shape.push_back(s);
            }
            i++;
        }
        else {
            new_shape.push_back(s);
        }
        j++;
    }
    for (uint32_t k = j; k < t1.get_dims(); ++k)
        new_shape.push_back(t1.get_shape(k));

    t2.reshape(new_shape);

    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = {0}[i];", t1, t2);
    k_runtime->make_job<unary_parameter>("squeeze", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void unsqueeze(uint32_t axis, const tensor& t1, tensor& t2) {
    std::vector<uint32_t> new_shape;
    uint32_t shape;
    for (uint32_t i = 0; i < axis; ++i) {
        shape = t1.get_shape(i);
        new_shape.emplace_back(shape);
    }
    new_shape.emplace_back(1);
    for (uint32_t i = axis; i < t1.get_dims(); ++i) {
        shape = t1.get_shape(i);
        new_shape.emplace_back(shape);
    }

    t2.reshape(new_shape);
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = {0}[i];", t1, t2);
    k_runtime->make_job<unary_parameter>("unsqueeze", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void clone(const tensor& t1, tensor& t2) {
    std::vector<uint32_t> new_shape;
    for (uint32_t i = 0; i < t1.get_dims(); ++i)
        new_shape.emplace_back(t1.get_shape(i));

    t2 = tensor(new_shape, t1.get_type());

    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = {0}[i];", t1, t2);
    k_runtime->make_job<unary_parameter>("clone", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void identity(const tensor& t1, const tensor& t2) {
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = {0}[i];", t1, t2);
    k_runtime->make_job<unary_parameter>("clone", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void isinf(const tensor& t1, const tensor& t2) {
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = isinf({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("isinf", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}

inline void isnan(const tensor& t1, const tensor& t2) {
    const unary_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = isnan({0}[i]);", t1, t2);
    k_runtime->make_job<unary_parameter>("isnan", kernel_code, { t1.get_data(), t2.get_data() }, p,
        set_group_size(p));
}


inline void logical_not(const tensor& t1, const tensor& t2)
{
    const logic_parameter p{ static_cast<uint32_t>(t1.get_size()) };
    if (t1.get_type() != BOOL || t2.get_type() != BOOL)
        throw std::runtime_error("Incorrect return type");

    const std::string kernel_code = unary_shader_code(unary_shader, "{1}[i] = !{0}[i]", t1, t2);
    k_runtime->make_job<logic_parameter>("or", kernel_code, { t1.get_data(), t2.get_data() },
        p, set_group_size(p));
}
