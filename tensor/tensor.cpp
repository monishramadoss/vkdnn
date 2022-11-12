// tensor.cpp : Source file for your target.
//

#include "tensor.h"
#include <vector>
#include <memory>
#include <cstring>
#include <iostream>

size_t k_tensor_count = 0;

view::view(const uint32_t* shape, const uint32_t dims, char data_size): n_dims_(dims), data_size_(data_size),
                                                                        shape_(new uint32_t[dims]),
                                                                        size_(new size_t[dims + 1]),
                                                                        stride_(new size_t[dims]),
                                                                        n_offsets_(0),
                                                                        offset_(nullptr)
{
	stride_[0] = 1;
	size_[n_dims_] = 1;

	for (uint32_t i = 0; i < n_dims_; ++i)
		shape_[i] = shape[i];

	for (uint32_t i = 1; i < n_dims_; ++i)
		stride_[i] = stride_[i - 1] * shape_[i - 1];

	for (uint32_t i = n_dims_ - 1u;;)
	{
		size_[i] = size_[i + 1u] * shape_[i];
		if (i == 0)
			break;
		--i;
	}

	for (uint32_t i = 0, j = n_dims_ - 1u; i < n_dims_ / 2u; ++i, --j)
	{
		const size_t tmp = stride_[i];
		stride_[i] = stride_[j];
		stride_[j] = tmp;
	}
}

view::view(uint32_t* shape, const uint32_t dims, const char data_size) : n_dims_(dims), data_size_(data_size),
                                                                         shape_(new uint32_t[dims]),
                                                                         size_(new size_t[dims + 1]),
                                                                         stride_(new size_t[dims]),
                                                                         n_offsets_(0),
                                                                         offset_(nullptr)

{
	stride_[0] = 1;
	size_[n_dims_] = 1;

	for (uint32_t i = 0; i < n_dims_; ++i)
		shape_[i] = shape[i];

	for (uint32_t i = 1; i < n_dims_; ++i)
		stride_[i] = stride_[i - 1] * shape_[i - 1];

	for (uint32_t i = n_dims_ - 1u;;)
	{
		size_[i] = size_[i + 1u] * shape_[i];
		if (i == 0)
			break;
		--i;
	}

	for (uint32_t i = 0, j = n_dims_ - 1u; i < n_dims_ / 2u; ++i, --j)
	{
		const size_t tmp = stride_[i];
		stride_[i] = stride_[j];
		stride_[j] = tmp;
	}
}


view::~view()
{
	delete[] shape_;
	delete[] size_;
	delete[] stride_;
	for (uint32_t i = 0; i < n_offsets_; ++i)
		delete[] offset_[i];
	delete[] offset_;
}

view::view(const view& v) : n_dims_(v.n_dims_), data_size_(v.data_size_), shape_(new uint32_t[n_dims_]),
                            size_(new size_t[n_dims_ + 1]), stride_(new size_t[n_dims_]),
                            n_offsets_(v.n_offsets_), offset_(new offset*[n_offsets_])
{
	std::memcpy(shape_, v.shape_, n_dims_ * sizeof(uint32_t));
	std::memcpy(stride_, v.stride_, n_dims_ * sizeof(uint32_t));
	std::memcpy(offset_, v.offset_, n_offsets_ * sizeof(offset*));
	std::memcpy(size_, v.size_, (1llu + n_dims_) * sizeof(uint32_t));
}

view::view(view&& v) noexcept : n_dims_(v.n_dims_), data_size_(v.data_size_), shape_(new uint32_t[n_dims_]),
                                size_(new size_t[n_dims_ + 1]), stride_(new size_t[n_dims_]),
                                n_offsets_(v.n_offsets_), offset_(new offset*[n_offsets_])
{
	std::memcpy(shape_, v.shape_, sizeof v.shape_);
	std::memcpy(stride_, v.stride_, sizeof v.stride_);
	if (n_offsets_ != 0)
		std::memcpy(offset_, v.offset_, sizeof v.offset_);
	std::memcpy(size_, v.size_, sizeof v.size_);
}

view& view::operator=(view&& v) noexcept
{
	if (this != &v && &v != nullptr)
	{
		n_dims_ = v.n_dims_;
		data_size_ = v.data_size_;
		shape_ = new uint32_t[n_dims_];
		size_ = new size_t[n_dims_ + 1];
		stride_ = new size_t[n_dims_];
		n_offsets_ = v.n_offsets_;
		offset_ = new offset*[n_offsets_];

		std::memcpy(shape_, v.shape_, sizeof v.shape_);
		std::memcpy(stride_, v.stride_, sizeof v.stride_);
		if (n_offsets_ != 0)
			std::memcpy(offset_, v.offset_, sizeof v.offset_);
		std::memcpy(size_, v.size_, sizeof v.size_);
	}
	return *this;
}

uint32_t view::ndims() const { return n_dims_; }

size_t view::size(const uint32_t idx) const { return size_[idx]; }

uint32_t view::shape(const uint32_t idx) const { return shape_[idx]; }

size_t view::bytes_length() const { return size_[0] * data_size_; }

size_t view::count(const uint32_t start_axis) const { return count(start_axis, n_dims_); }


char get_data_size(const DTYPE type)
{
	if (type == DOUBLE || type == LINT || type == LUINT)
		return 8;
	if (type == FLOAT || type == INT || type == UINT || type == BOOL)
		return 4;
	if (type == HFLOAT || type == HINT || type == HUINT)
		return 2;
	if (type == INT8 || type == UINT8)
		return 1;
	return 0;
}


size_t view::count(const uint32_t start_axis, uint32_t end_axis) const
{
	if (end_axis > n_dims_)
		end_axis = n_dims_;
	if (start_axis == end_axis)
		return 0;
	uint32_t acc = 1;
	for (auto i = start_axis; i < end_axis; ++i)
		acc *= shape_[i];
	return acc;
}

view view::index(const uint32_t idx, int dim)
{
	if (shape_[dim] < idx)
		std::cerr << "UNABLE TO SPLIT ACCROSS IDX" << std::endl;

	if (dim == -1)
		dim = n_dims_ - 1l;

	const size_t upper_split_size = dim == 0 ? 1 : count(0, dim);
	const size_t lower_split_size = count(dim, -1) * data_size_;

	const size_t split_stride = stride_[dim];
	const size_t off = split_stride * idx * data_size_;

	auto* dst_shape = new uint32_t[n_dims_];
	memcpy(dst_shape, shape_, n_dims_);
	dst_shape[dim] = 1;


	if (const auto new_offset = static_cast<offset**>(realloc(offset_, sizeof(offset*) * (n_offsets_ + 1))); new_offset
		!= nullptr)
		offset_ = new_offset;
	offset_[n_offsets_] = new offset[upper_split_size];
	for (uint32_t i = 0; i < upper_split_size; ++i)
		offset_[n_offsets_][i] = {off + i * lower_split_size, split_stride};
	++n_offsets_;
	return {dst_shape, n_dims_, data_size_};
}

view view::reshape(const uint32_t* shape, uint32_t dims)
{
	size_t acc = shape[0];
	for (uint32_t i = 1; i < dims; ++i)
		acc *= shape[i];
	if (acc != size_[0])
		throw std::runtime_error("shape error");
	return {shape, dims, data_size_};
}


tensor::tensor(std::vector<uint32_t>& shape, const DTYPE type) : view_(shape.data(), shape.size(), get_data_size(type)),
                                                                 data_(new vk_block*[2]), parent_(nullptr),
                                                                 d_type_(type),
                                                                 name_("tensor_" + std::to_string(++k_tensor_count))
{
	data_[0] = k_runtime->malloc(view_.bytes_length(), false);
	data_[1] = nullptr;
}


tensor::tensor(const std::vector<uint32_t>& shape, const DTYPE type) : view_(shape.data(), shape.size(),
                                                                             get_data_size(type)),
                                                                       data_(new vk_block*[2]), parent_(nullptr),
                                                                       d_type_(type),
                                                                       name_("tensor_" + std::to_string(
	                                                                       ++k_tensor_count))
{
	data_[0] = k_runtime->malloc(view_.bytes_length(), false);
	data_[1] = nullptr;
}


tensor::tensor(tensor* ptr, view v) : view_(std::move(v)), data_(nullptr), parent_(ptr), d_type_(ptr->d_type_),
                                      name_("tensor_" + std::to_string(++k_tensor_count))
{
}


tensor tensor::index(const uint32_t i, const int dim)
{
	return tensor(this, view_.index(i, dim));
}

tensor& tensor::reshape(const std::vector<uint32_t>& shape)
{
	view_ = view_.reshape(shape.data(), static_cast<uint32_t>(shape.size()));
	return *this;
}

tensor& tensor::reshape(std::vector<uint32_t>& shape)
{
	view_ = view_.reshape(shape.data(), static_cast<uint32_t>(shape.size()));
	return *this;
}      

tensor::tensor(tensor&& t) noexcept : view_(std::move(t.view_)), data_(t.data_), parent_(t.parent_), d_type_(t.d_type_),
                                      name_(std::move(t.name_))
{
	t.data_ = nullptr;
}

tensor& tensor::operator=(tensor&& t) noexcept
{
	if (&t != this)
	{
		view_ = std::move(t.view_);
		data_ = t.data_;
		parent_ = t.parent_;
		d_type_ = t.d_type_;
		name_ = std::move(t.name_);
	}
	return *this;
}


tensor::~tensor()
{
	if (data_ != nullptr)
	{
		k_runtime->free(data_[0]);
		if (data_[1] != nullptr)
			k_runtime->free(data_[1]);
	}
}

vk_block* tensor::get_data() const
{
	return data_[0];
}

vk_block* tensor::get_host_data() const
{
	if (data_[1] == nullptr)
	{
		data_[1] = k_runtime->malloc(data_[0]->size, true);
		memset(data_[1]->ptr, 0, data_[1]->size);
	}
	return data_[1];
}


void tensor::sync(const bool to_device) const
{
	if (data_[1] == nullptr)
	{
		data_[1] = k_runtime->malloc(data_[0]->size, true);
		memset(data_[1]->ptr, 0, data_[1]->size);
	}

	if (to_device)
		k_runtime->memcpy(data_[1], data_[0], 0, 0);
	else
		k_runtime->memcpy(data_[0], data_[1], 0, 0);
}

void tensor::set_data(vk_block* blk)
{
	data_ = &blk;
}



int gen_type(const DTYPE type, std::string& type_name)
{
	if (type == BOOL)
	{
		type_name = "bool";
		return 0;
	}
	if (type == HFLOAT)
	{
		type_name = "float16_t";
		return 5;
	}
	if (type == FLOAT)
	{
		type_name = "float";
		return 6;
	}
	if (type == DOUBLE)
	{
		type_name = "float64_t";
		return 7;
	}
	if (type == INT8)
	{
		type_name = "int8_t";
		return 1;
	}
	if (type == HINT)
	{
		type_name = "int16_t";
		return 2;
	}
	if (type == INT)
	{
		type_name = "int32_t";
		return 3;
	}
	if (type == LINT)
	{
		type_name = "int64_t";
		return 4;
	}
	if (type == UINT8)
	{
		type_name = "uint8_t";
		return 1;
	}
	if (type == HUINT)
	{
		type_name = "uint16_t";
		return 2;
	}
	if (type == UINT)
	{
		type_name = "uint32_t";
		return 3;
	}
	if (type == LUINT)
	{
		type_name = "uint64_t";
		return 4;
	}
	if (type == NONE)
	{
		type_name = "";
		return -1;
	}
	return -1;
}

int tensor_injection(std::string& body, std::string& var_name, const int i, const tensor& t1)
{
	const DTYPE type = t1.get_type();
	var_name = "tensor_" + std::to_string(i);
	std::string type_name;
	const int ext_id = gen_type(t1.get_type(), type_name);
	const std::string buffer_layout = "layout(binding=" + std::to_string(i) + ") buffer buf_" + std::to_string(i) +
		" { " +	type_name + " " + var_name + "[]; };\n";
	body += buffer_layout;
	return ext_id;
}
