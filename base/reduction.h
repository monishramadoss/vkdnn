// reduction.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

#define THREAD_COUNT 128

inline std::string reduction_shader_code(const std::string& kernel_shader_code, const std::string fn_pass, const tensor& t1, const tensor& t2)
{
    std::string local_shader = "#define THREAD_COUNT " + std::to_string(THREAD_COUNT) + kernel_shader_code;
    std::string shader_tensor[2];
    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, t1);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t2);
    local_shader = SHADER_VERSION + local_shader;
    std::string tmp_type;
    gen_type(t2.get_type(), tmp_type);
    std::string fmt_str = Format(fn_pass, "val", "val_op");

    local_shader += R"(
shared )" + tmp_type + R"( thread_values[THREAD_COUNT];
void reduce_op()" + tmp_type + R"( val, inout )" + tmp_type + R"( val_op) {
    {0}
}

void reduce(uint tid){
    reduce_op(thread_values[tid + 64], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 32], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 16], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 8], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 4], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 2], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 1], thread_values[tid]);
    memoryBarrierShared();
}

void main() {        
    uint tid = gl_LocalInvocationID.x;
    uint nwg = gl_NumWorkGroups.x;
    uint wgid = gl_WorkGroupID.x;
    uint blkS = gl_WorkGroupSize.x;
    
    for(uint row = wgid; row < rows; row += nwg){
        thread_values[tid] =  )" + tmp_type + R"((0);
        
        for(uint col = tid; col < cols; col += blkS)
            reduce_op()" + shader_tensor[0] + R"([col * rows + row], thread_values[tid]);
        reduce(tid);
        if(tid == 0)
            )" + shader_tensor[1] + R"([row] = thread_values[0];
    }
})";

    local_shader = Format(local_shader, fmt_str);
    return local_shader;
}

inline std::string reduction_kernel_code = R"(
layout(push_constant) uniform pushBlock {
    uint rows;
    uint cols;
    uint total;
};

layout(local_size_x = THREAD_COUNT) in; 

)";

struct reduction_param
{
    uint32_t rows;
    uint32_t cols;
    uint32_t total;
};

inline uint32_t set_group_size(reduction_param p)
{
    return align_size(p.total, 32) / 32;
}

void reduce_sum(const int axis, const tensor& t1, const tensor& t2)
{
    const reduction_param p{ t1.get_size(axis), t1.get_size(0, axis), t1.get_size() };
    std::string kernel_code = reduction_shader_code(reduction_kernel_code, "{1} += {0};", t1, t2);
    k_runtime->make_job<reduction_param>("reduce_sum", kernel_code, { t1.get_data(), t2.get_data()
        }, p, set_group_size(p));
}

void reduce_mul(const int axis, const tensor& t1, const tensor& t2)
{
    const reduction_param p{ t1.get_size(axis), t1.get_size(0, axis), t1.get_size() };
    std::string kernel_code = reduction_shader_code(reduction_kernel_code, "{1} *= {0};", t1, t2);
    k_runtime->make_job<reduction_param>("reduce_mul", kernel_code, { t1.get_data(), t2.get_data()
        }, p, set_group_size(p));
}

void redcue_min(const int axis, const tensor& t1, const tensor& t2)
{
    const reduction_param p{ t1.get_size(axis), t1.get_size(0, axis), t1.get_size() };
    std::string kernel_code = reduction_shader_code(reduction_kernel_code, "{1} = min({0}, {1});", t1, t2);
    k_runtime->make_job<reduction_param>("reduce_min", kernel_code, { t1.get_data(), t2.get_data()
        }, p, set_group_size(p));
}

void reduce_max(const int axis, const tensor& t1, const tensor& t2)
{
    const reduction_param p{ t1.get_size(axis), t1.get_size(0, axis), t1.get_size() };
    std::string kernel_code = reduction_shader_code(reduction_kernel_code, "{1} = max({0}, {1});", t1, t2);
    k_runtime->make_job<reduction_param>("reduce_max", kernel_code, { t1.get_data(), t2.get_data()
        }, p, set_group_size(p));
}


inline std::string reduce_mean_shader_code(const std::string& kernel_shader_code, const tensor& t1, const tensor& t2)
{
    std::string local_shader = "#define THREAD_COUNT " + std::to_string(THREAD_COUNT) + kernel_shader_code;
    std::string shader_tensor[2];
    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, t1);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t2);
    local_shader = SHADER_VERSION + local_shader;
    std::string tmp_type;
    gen_type(t2.get_type(), tmp_type);

    local_shader += R"(
shared )" + tmp_type + R"( thread_values[THREAD_COUNT];
void reduce_op()" + tmp_type + R"( val, inout )" + tmp_type + R"( val_op) {
    val_op += val;
}

void reduce(uint tid){
    reduce_op(thread_values[tid + 64], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 32], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 16], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 8], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 4], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 2], thread_values[tid]);
    memoryBarrierShared();
    reduce_op(thread_values[tid + 1], thread_values[tid]);
    memoryBarrierShared();
}

void main() {        
    uint tid = gl_LocalInvocationID.x;
    uint nwg = gl_NumWorkGroups.x;
    uint wgid = gl_WorkGroupID.x;
    uint blkS = gl_WorkGroupSize.x;
    
    for(uint row = wgid; row < rows; row += nwg){
        thread_values[tid] = )" + tmp_type + R"((0);
        
        for(uint col = tid; col < cols; col += blkS)
            reduce_op()" + shader_tensor[0] + R"([col * rows + row], thread_values[tid]);
        reduce(tid);
        if(tid == 0)
            )" + shader_tensor[1] + R"([row] = thread_values[0] / cols;
    }
})";

    return local_shader;
}

void reduce_mean(const int axis, const tensor& t1, const tensor& t2)
{
    const reduction_param p{t1.get_size(axis), t1.get_size(0, axis), t1.get_size() };
    std::string kernel_code = reduce_mean_shader_code(reduction_kernel_code, t1, t2);
    k_runtime->make_job<reduction_param>("reduce_mean", kernel_code, { t1.get_data(), t2.get_data()
        }, p, set_group_size(p));
}


