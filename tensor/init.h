// init.h : Header file for your target.

#pragma once
#include "utils.h"

template <class T>
tensor constant_t(const std::vector<uint32_t>& shape, T v)
{
	throw std::runtime_error("NOT IMPLEMENTED");
}

template <class T>
tensor arange_t(T low, T high, T step)
{
	throw std::runtime_error("NOT IMPLEMENTED");
}

template <class T>
tensor arange_t(T high) { return arange_t<T>(static_cast<T>(0), high, static_cast<T>(1)); }

template <class T>
tensor zeros(const std::vector<uint32_t>& shape) { return constant_t<T>(shape, 0); }

template <class T>
tensor ones(const std::vector<uint32_t>& shape) { return constant_t<T>(shape, 1); }

struct fill_param
{
	uint32_t total;
};

inline uint32_t set_group_size(const fill_param& p)
{
	return align_size(p.total, 1024) / 1024;
}

inline std::string fill_shader =
	"layout(push_constant) uniform pushBlock {\n\t uint total;\n};\nlayout(local_size_x=1024, local_size_y=1, local_size_z=1) in;\n";


template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const int8_t v)
{
	tensor t(shape, INT8);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = int8_t(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_int8", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}

template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const uint8_t v)
{
	tensor t(shape, UINT8);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = uint8_t(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_uint8", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const int16_t v)
{
	tensor t(shape, HINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = int16_t(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_int16", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const uint16_t v)
{
	tensor t(shape, HUINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = uint16_t(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_uint16", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const int v)
{
	tensor t(shape, INT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = int(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_int", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const uint32_t v)
{
	tensor t(shape, UINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = uint(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_uint", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const int64_t v)
{
	tensor t(shape, LINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = int64_t(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_int64", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const uint64_t v)
{
	tensor t(shape, LUINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = uint64_t(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_uint64", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const float v)
{
	tensor t(shape, FLOAT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = float(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_float", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor constant_t(const std::vector<uint32_t>& shape, const double v)
{
	tensor t(shape, DOUBLE);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader, "{}[i] = float64_t(" + std::to_string(v) + ");", t);
	k_runtime->make_job<fill_param>("constant_double", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor arange_t(const int8_t low, const int8_t high, const int8_t step)
{
	uint32_t size = (high - low) / step;
	tensor t({size}, INT8);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = int8_t(i * " + std::to_string(step) + " + " +
	                                                std::to_string(low) + ");", t);
	k_runtime->make_job<fill_param>("arange_int8", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}

template <>
inline tensor arange_t(const uint8_t low, const uint8_t high, const uint8_t step)
{
	uint32_t size = (high - low) / step;
	tensor t({size}, UINT8);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = uint8_t(* i * " + std::to_string(step) + " + " +
	                                                std::to_string(low) + ");", t);
	k_runtime->make_job<fill_param>("arange_uint8", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor arange_t(const int16_t low, const int16_t high, const int16_t step)
{
	uint32_t size = (high - low) / step;
	tensor t({size}, HINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = int16_t(i * " + std::to_string(step) + " + " +
	                                                std::to_string(low) + ");", t);
	k_runtime->make_job<fill_param>("fill_int16", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor arange_t(const uint16_t low, const uint16_t high, const uint16_t step)
{
	uint32_t size = (high - low) / step;
	tensor t({size}, HUINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = uint16_t(i * " + std::to_string(step) + " + " +
	                                                std::to_string(low) + ");", t);
	k_runtime->make_job<fill_param>("arange_uint16", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor arange_t(const int low, const int high, const int step)
{
	uint32_t size = (high - low) / step;
	tensor t({size}, INT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = int(i * " + std::to_string(step) + " + " + std::to_string(
		                                                low) + ");", t);
	k_runtime->make_job<fill_param>("arange_int", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor arange_t(const uint32_t low, const uint32_t high, const uint32_t step)
{
	uint32_t size = (high - low) / step;
	tensor t({size}, UINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = uint(i * " + std::to_string(step) + " + " + std::to_string(
		                                                low) + ");", t);
	k_runtime->make_job<fill_param>("arange_uint", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor arange_t(const int64_t low, const int64_t high, const int64_t step)
{
	auto size = static_cast<uint32_t>((high - low) / step);
	tensor t({size}, LINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = int64_t(i * " + std::to_string(step) + " + " +
	                                                std::to_string(low) + ");", t);
	k_runtime->make_job<fill_param>("arange_int64", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor arange_t(const uint64_t low, const uint64_t high, const uint64_t step)
{
	auto size = static_cast<uint32_t>((high - low) / step);
	tensor t({size}, LUINT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = uint64_t(i * " + std::to_string(step) + " + " +
	                                                std::to_string(low) + ");", t);
	k_runtime->make_job<fill_param>("arange_uint64", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


#include <iostream>

template <>
inline tensor arange_t(const float low, const float high, const float step)
{
	auto size = static_cast<uint32_t>(std::ceilf((high - low) / step));
	tensor t({size}, FLOAT);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = float(i * " + std::to_string(step) + " + " +
	                                                std::to_string(low) + ");", t);
	k_runtime->make_job<fill_param>("arange_float", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}


template <>
inline tensor arange_t(const double low, const double high, const double step)
{
	auto size = static_cast<uint32_t>(std::ceil((high - low) / step));
	tensor t({size}, DOUBLE);
	const fill_param p{static_cast<uint32_t>(t.get_size())};
	const std::string kernel = singlton_shader_code(fill_shader,
	                                                "{}[i] = float64_t(i * " + std::to_string(step) + " + " +
	                                                std::to_string(low) + ");", t);
	k_runtime->make_job<fill_param>("arange_double", kernel, {t.get_data()}, p, set_group_size(p));
	return t;
}
