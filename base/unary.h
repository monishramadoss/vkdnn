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

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = sin(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("sin", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void asin(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = asin(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("asin", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void cos(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = cos(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("cos", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void acos(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = acos(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("acos", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void tan(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = tan(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("tan", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void atan(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = atan(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("atan", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void sinh(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = sinh(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("sinh", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void asinh(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = asinh(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("asinh", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void cosh(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = cosh(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("cosh", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void acosh(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = acosh(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("acosh", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void tanh(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = tanh(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("tanh", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void atanh(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = atanh(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("atanh", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}


inline void neg(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = -%s[i];", t1, t2);
	k_runtime->make_job<unary_parameter>("neg", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void inv(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = 1/%s[i];", t1, t2);
	k_runtime->make_job<unary_parameter>("inv", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void log(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = log(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("log", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void log2(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = log2(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("log2", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void exp(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = exp(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("exp", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void exp2(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = exp2(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("exp2", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void sqrt(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = sqrt(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("sqrt", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void isqrt(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = inversesqrt(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("inv_sqrt", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}


inline void increment(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = ++%s[i];", t1, t2);
	k_runtime->make_job<unary_parameter>("increment", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void decrement(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = --%s[i];", t1, t2);
	k_runtime->make_job<unary_parameter>("decrement", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void ceil(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
	if (const auto t2_type = t2.get_type(); t2_type == FLOAT || t2_type == DOUBLE || t2_type == BOOL || t2_type ==
		HFLOAT || t2_type == NONE)
		throw std::runtime_error("Return time is not correct");

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = ceil(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("ceil", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}

inline void floor(const tensor& t1, const tensor& t2)
{
	const unary_parameter p{static_cast<uint32_t>(t1.get_size())};
	if (const auto t2_type = t2.get_type(); t2_type == FLOAT || t2_type == DOUBLE || t2_type == BOOL || t2_type ==
		HFLOAT || t2_type == NONE)
		throw std::runtime_error("Return time is not correct");

	const std::string kernel_code = unary_shader_code(unary_shader, "%s[i] = floor(%s[i]);", t1, t2);
	k_runtime->make_job<unary_parameter>("floor", kernel_code, {t1.get_data(), t2.get_data()}, p,
	                                                 set_group_size(p));
}
