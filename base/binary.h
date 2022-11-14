// binary.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

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

inline void add(const tensor& t1, const tensor& t2, const tensor& t3)
{
	const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
	const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] + {1}[i];", t1, t2, t3);
	k_runtime->make_job<binary_parameter>("add", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
	                                                  p, set_group_size(p));
}

inline void sub(const tensor& t1, const tensor& t2, const tensor& t3)
{
	const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
	const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] - {1}[i];", t1, t2, t3);
	k_runtime->make_job<binary_parameter>("sub", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
	                                                  p, set_group_size(p));
}

inline void mul(const tensor& t1, const tensor& t2, const tensor& t3)
{
	const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
	const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] * {1}[i];", t1, t2, t3);
	k_runtime->make_job<binary_parameter>("mul", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
	                                                  p, set_group_size(p));
}

inline void true_div(const tensor& t1, const tensor& t2, const tensor& t3)
{
	const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
	const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = {0}[i] / {1}[i];", t1, t2, t3);
	k_runtime->make_job<binary_parameter>("div", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
	                                                  p, set_group_size(p));
}

inline void mod(const tensor& t1, const tensor& t2, const tensor& t3)
{
	const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
	const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = mod({0}[i], {1}[i]);", t1, t2, t3);
	k_runtime->make_job<binary_parameter>("mod", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
	                                                  p, set_group_size(p));
}

inline void pow(const tensor& t1, const tensor& t2, const tensor& t3)
{
	const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
	const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = pow({0}[i], {1}[i]);", t1, t2, t3);
	k_runtime->make_job<binary_parameter>("pow", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
	                                                  p, set_group_size(p));
}

inline void max(const tensor& t1, const tensor& t2, const tensor& t3)
{
	const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
	const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = max({0}[i], {1}[i]);", t1, t2, t3);
	k_runtime->make_job<binary_parameter>("max", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
	                                                  p, set_group_size(p));
}

inline void min(const tensor& t1, const tensor& t2, const tensor& t3)
{
	const binary_parameter p{static_cast<uint32_t>(t1.get_size())};
	const std::string kernel_code = binary_shader_code(binary_shader, "{2}[i] = min({0}[i], {1}[i]);", t1, t2, t3);
	k_runtime->make_job<binary_parameter>("min", kernel_code, {t1.get_data(), t2.get_data(), t3.get_data()},
	                                                  p, set_group_size(p));
}
