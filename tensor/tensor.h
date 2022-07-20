﻿// tensor.h : Header file for your target.

#pragma once
#include <vector>
#include "../runtime/runtime.h"


struct offset
{
	size_t upper;
	size_t lower;
};


class view
{
	size_t mNDims;
	size_t mDataSize{};

	size_t* mShape;
	size_t* mSize;
	size_t* mStride;
	offset** mOffset;
public:
	view(const size_t* shape, size_t dims, char data_size);
	~view();
	size_t ndims() const { return mNDims; }
	size_t size(size_t idx = 0) const { return mSize[idx]; }
	size_t shape(size_t idx = 0) const { return mShape[idx]; }
	size_t bytes_length() const { return mSize[0] * mDataSize; }

	size_t count(size_t start_axis = 0) const { return count(start_axis, mNDims); }
	size_t count(size_t start_axis, size_t end_axis) const;

	view operator[] (size_t idx);
};


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


class tensor final
{
	view m_view;
	vk_block** m_data{};
	tensor* m_parent{};
	DTYPE m_dType{};
public:
	explicit tensor(std::vector<size_t>& shape, DTYPE type = FLOAT);
	explicit tensor(const std::vector<size_t>& shape, DTYPE type = FLOAT);
	~tensor();
	DTYPE getType() const { return m_dType; }
	vk_block* get_data() const;
	size_t get_size(size_t i=0) const { return m_view.size(i); }
};

