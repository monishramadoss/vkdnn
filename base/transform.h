// transform.h : Header file for your target

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"


inline std::string permute_kernel_code(const std::string& kernel_shader_code, const tensor& pt, const tensor& t1, const tensor& t2)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[3];
    parameter_shader_code(shader_tensor, local_shader, pt, t1, t2);

    local_shader += R"(
void main() {
    for(uint i = gl_GlobalInvocationID.x; i < total; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x){
        uint old_pos = 0;
        uint new_pos = i;
        for(uint j = 0; j < ndims; ++j){
            uint order = )" + shader_tensor[0] + R"([j];
            old_pos += (new_pos / )" + shader_tensor[0] + R"([ndims+j]) * )" + shader_tensor[0] + R"([ndims*2 + order];
            new_pos %= )" + shader_tensor[0] + R"([ndims+j];
        }
        )" + shader_tensor[2] + R"([i] = )" + shader_tensor[1] + R"([abs(old_pos)];
    }
}
)";

    return local_shader;
}

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

struct permute_param
{
    uint32_t total;
    uint32_t ndims;
};

inline uint32_t set_group_size(permute_param p)
{
    return align_size(p.total, 1024) / 1024;
}

void permute(const std::vector<size_t>& axes, const tensor& t1, const tensor& t2, tensor& param_tensor)
{
    const permute_param p{ static_cast<uint32_t>(t1.get_size()), t1.get_dims() };
    if(param_tensor.is_empty())
    {
        param_tensor = tensor({ t1.get_dims() * 2 + 1 }, UINT);
        void* stride = param_tensor.get_host_data();
    }
    const std::string kernel_code = permute_kernel_code(parameter_kernel_code, param_tensor, t1, t2);

    k_runtime->make_job<permute_param>("permute", kernel_code, { param_tensor.get_data(), t1.get_data(), t2.get_data() },
        p, set_group_size(p));
}

inline std::string concat_kernel_code = R"(
layout(push_constant) uniform pushBlock {
    uint total;
    uint concat_size;
    uint acc_concat_axis;
    uint out_concat_axis;
    uint total_concat_size;
};

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
)";

inline std::string concat_shader_code(const std::string& kernel_shader_code, const tensor& t1, const tensor& t2) {
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[2];

    const std::string shader_code = R"(
        uint concat_num = i / total_concat_size;
        uint concat_idx = i % total_concat_size;
        uint out_idx = concat_idx  + (concat_num * out_concat_axis + acc_concat_axis) * concat_size;
        {1}[out_idx] = {0}[i];
)";
    unary_shader_code(local_shader, shader_code, t1, t2);
    return local_shader;
}


struct concat_param
{
    uint32_t total;
    uint32_t concat_size;
    uint32_t acc_concat_axis;
    uint32_t out_concat_axis;
    uint32_t total_concat_size;
};

inline uint32_t set_group_size(concat_param p)
{
    return align_size(p.total, 1024) / 1024;
}

void concat(uint32_t axis, const std::vector<tensor>& tn, tensor& t2) {
    tensor t1 = tensor({ 0 });// = tn.front();
    uint32_t sum_axis = t1.get_shape(axis);
    uint32_t dim_num = t1.get_dims();
    uint32_t concat_size = t2.get_size(axis + 1);
    uint32_t acc_concat_axis = 0;

    for (size_t i = 1; i < tn.size(); ++i) 
        sum_axis += tn[i].get_shape(axis);

    for (size_t i = 1; i < tn.size(); ++i) {
        tensor ti = tensor({ 0 });// = tn[i];
        uint32_t total_concat_size = ti.get_size(axis);
        const concat_param p{ static_cast<uint32_t>(ti.get_size()), concat_size, acc_concat_axis, sum_axis, total_concat_size };
        acc_concat_axis += ti.get_shape(axis);
        const std::string kernel_code = concat_shader_code(" ", ti, t2);
        k_runtime->make_job<concat_param>("concat", kernel_code, { ti.get_data(), t2.get_data() }, p, set_group_size(p));
    }


}