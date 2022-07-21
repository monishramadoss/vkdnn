// tensor.cpp : Source file for your target.
//

#include "tensor.h"


#include <vector>


view::view(const size_t* shape, size_t dims, char data_size): mNDims(dims), mShape(new size_t[dims]),
                                                              mSize(new size_t[dims + 1]), mStride(new size_t[dims]),
                                                              mOffset(nullptr)
{
	mDataSize =  data_size;
	mStride[0] = 1;
	mSize[mNDims] = 1;

	for(size_t i = 0; i < dims; ++i)
	{
		mShape[i] = shape[i];
		mDataSize *= shape[i];
	}

	for (size_t i = 1; i < mNDims; ++i)
		mStride[i] = mShape[i - 1] * mStride[i - 1];

	for (size_t i = mNDims - 1;;)
	{
		mSize[i] = mShape[i] * mSize[i + 1];
		if (i == 0)
			break;
		--i;
	}

	for (size_t i = 0, j = mNDims-1; i < mNDims/2; ++i, --j)
	{
		const size_t tmp = mStride[i];
		mStride[i] = mStride[j];
		mStride[j] = tmp;
	}
}

view::~view()
{
	delete[] mShape;
	delete[] mSize;
	delete[] mStride;
	delete[] mOffset;
}


char get_data_size(DTYPE type)
{
	if (type == DTYPE::DOUBLE || type == DTYPE::LINT || type == DTYPE::LUINT)
		return 8;
	else if (type == DTYPE::FLOAT || type == DTYPE::INT || type == DTYPE::UINT || type == DTYPE::BOOL)
		return 4;
	else if (type == DTYPE::HFLOAT || type == DTYPE::HINT || type == DTYPE::HUINT)
		return 2;
	else if (type == DTYPE::INT8 || type == DTYPE::UINT8)
		return 1;
	else
		return 0;
}


size_t view::count(size_t start_axis, size_t end_axis) const
{
	if (start_axis == end_axis)
		return 0;
	size_t acc = 1;
	for (auto i = start_axis; i < end_axis; ++i)
		acc *= mShape[i];
	return acc;
}

tensor::tensor(std::vector<size_t>& shape, DTYPE type) : m_view(shape.data(), shape.size(), get_data_size(type)), m_dType(type)
{
	m_data = kRuntime.malloc(m_view.bytes_length());
}

tensor::tensor(const std::vector<size_t>& shape, const DTYPE type) : m_view(shape.data(), shape.size(), get_data_size(type)), m_dType(type)
{
	m_data = kRuntime.malloc(m_view.bytes_length());
}

tensor::~tensor()
{
	kRuntime.free(m_data);
}

vk_block* tensor::get_data() const
{
	return *m_data;
}


