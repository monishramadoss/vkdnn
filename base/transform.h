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


struct folding_params
{
    uint32_t batch_size;
    uint32_t channel_size;
    uint32_t output_size;
    uint32_t kernel_size;
    uint32_t batch_stride;
    uint32_t channel_stride;
};  

inline std::string folding_kernel_code = R"(
layout(push_constant) uniform pushBlock {
    uint batch_size;
    uint channel_size;
    uint output_size;
    uint kernel_size;
    uint batch_stride;
    uint channel_stride;
};

layout(local_size_x = 256, local_size_y = 4, local_size_z = 1) in;
)";

inline uint32_t set_group_size_x(folding_params p)
{
    return align_size(p.batch_size * p.output_size, 256) / 256;
}

inline uint32_t set_group_size_y(folding_params p)
{
    return align_size(p.channel_size * p.kernel_size, 4) / 4;
}

template<uint32_t ndims>
void unfold(tensor& t1, tensor& t2, tensor& param_tensor)
{
    if(param_tensor.is_empty())
    {
        param_tensor = tensor({ ndims * 4 });
    }
    std::string kernel_code = folding_kernel_code;
    kernel_code += "uint ndims = " + std::to_string(ndims) + ";\n";
    kernel_code = unfold_kernel_code(kernel_code, t1, t2, param_tensor);
    const folding_params p = {
        .batch_size = t1.get_shape(0),
        .channel_size = t1.get_shape(1),
        .output_size = static_cast<uint32_t>(t2.get_size(2)),
        .kernel_size = t2.get_shape(1),
        .batch_stride = static_cast<uint32_t>(t1.get_size(0)),
        .channel_stride = static_cast<uint32_t>(t1.get_size(2)),
    };
    k_runtime->make_job<folding_params>("unfold", kernel_code, { t1.get_data(), t2.get_data(), param_tensor.get_data() },
        p, set_group_size_x(p), set_group_size_y(p));
    
}

template<uint32_t ndims>
void fold(tensor& t1, tensor& t2, tensor& param_tensor)
{
    std::string kernel_code = folding_kernel_code;
    std::string kernel_code = folding_kernel_code;
    kernel_code += "uint ndims = " + std::to_string(ndims) + ";\n";
}