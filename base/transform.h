#pragma once
#include "../tensor/tensor.h"


uint32_t* prepareStrides(const std::vector<uint32_t>& shape_before, const std::vector<uint32_t>& shape_after, uint32_t* stride)
{
    size_t dims = shape_before.size();
    stride[2 * dims - 1] = 1;
    stride[3 * dims - 1] = 1;

    for (int64_t i = dims - 2; i >= 0; i--)
    {
        stride[dims * 2 + i] = stride[dims * 2 + i + 1] * shape_before[i + 1];
        stride[dims + i] = stride[dims + i + 1] * shape_after[i + 1];
    }
    return stride;
}


inline std::string parameter_kernel_code = R"(
layout(push_constant) uniform pushBlock {
    uint total;
    uint ndims;
};

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
)";

struct transpose_param
{
    uint32_t total;
    uint32_t ndims;
};

inline uint32_t set_group_size(transpose_param p)
{
    return align_size(p.total, 1024) / 1024;
}

void transpose(const std::vector<size_t>& axes, const tensor& t1, const tensor& t2, tensor& param_tensor)
{
    const transpose_param p{ static_cast<uint32_t>(t1.get_size()), t1.get_dims() };
    if(param_tensor.is_empty())
    {
        param_tensor = tensor({ t1.get_dims() * 2 + 1 }, UINT);
        void* stride = param_tensor.get_host_data();
    }
    const std::string kernel_code = transpose_kernel_code(parameter_kernel_code, param_tensor, t1, t2);

}


