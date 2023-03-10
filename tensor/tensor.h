// tensor.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "view.h"
#include <vector>
#include <string>

enum DTYPE
{
	NONE = 0,
	HFLOAT = 1,
	FLOAT = 2,
	DOUBLE = 3,
	HINT = 4,
	INT = 5,
	LINT = 6,
	HUINT = 7,
	UINT = 8,
	LUINT = 9,
	INT8 = 10,
	UINT8 = 11,
	BOOL = 12,
};

enum  CHUNK_PROTOCOL
{
	ROW_SPEC = 1,
	COL_SPEC = 2,
	ALL_SPEC = 3,
};

struct chunk_spec
{
	CHUNK_PROTOCOL spec;
	uint32_t dim;
};

class tensor final
{
	chunk_spec cspec_;
	view view_;
	vk_block** data_;
	tensor* parent_ = nullptr;
	DTYPE d_type_;
	std::string name_;
public:
    explicit tensor(std::vector<uint32_t>& shape, DTYPE type = FLOAT);
	explicit tensor(const std::vector<uint32_t>& shape={}, DTYPE type = FLOAT);
	explicit tensor(tensor* ptr, view v);
	~tensor();

	tensor index(uint32_t i, int dim);
	tensor& reshape(const std::vector<uint32_t>& shape);
	tensor& reshape(std::vector<uint32_t>& shape);


	tensor(tensor&& t) noexcept;
	tensor& operator=(tensor&& t) noexcept;

	[[nodiscard]] DTYPE get_type() const { return d_type_; }
	[[nodiscard]] vk_block* get_data() const;
	[[nodiscard]] vk_block* get_host_data() const;
	void sync(bool to_device = true) const;

	void set_data(vk_block*);
	[[nodiscard]] size_t get_size(const uint32_t i = 0) const { return view_.size(i); }
	[[nodiscard]] uint32_t get_dims() const { return view_.ndims(); }
	[[nodiscard]] uint32_t get_shape(const uint32_t i = 0) const { return view_.shape(i); }
	[[nodiscard]] size_t get_bytes_size() const { return view_.bytes_length(); }
	[[nodiscard]] bool is_empty() const { return get_bytes_size() == 0; }
	job* parent_job = nullptr;


	template <class T>
	static tensor fill(const std::vector<uint32_t>& shape, T t);
	template <class T>
	static tensor zeros(const std::vector<uint32_t>& shape);
	template <class T>
	static tensor ones(const std::vector<uint32_t>& shape);
};


inline const char* shader_extensions[]{
	"", // 0
	"#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable \n", // 1
	"#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable \n", // 2 
	"#extension GL_EXT_shader_explicit_arithmetic_types_int32: enable \n", // 3
	"#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable \n", // 4
	"#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable \n", // 5
	"#extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable \n", // 6
	"#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable \n" // 7
};


int gen_type(DTYPE type, std::string& type_name);


int tensor_injection(std::string& body, std::string& var_name, int i, const tensor& t1);


#include "init.h"


template <class T>
tensor tensor::fill(const std::vector<uint32_t>& shape, T t)
{
	return constant_t<T>(shape, t);
}


template <class T>
tensor tensor::zeros(const std::vector<uint32_t>& shape)
{
	return constant_t<T>(shape, 0);
}

template <class T>
tensor tensor::ones(const std::vector<uint32_t>& shape)
{
	return constant_t<T>(shape, 1);
}
