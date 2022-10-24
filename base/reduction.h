// reduction.h : Header file for your target.

#pragma once
#include "../tensor/tensor.h"

inline std::string reduction_shader = R"(
layout(push_constant) uniform pushBlock {
	uint total;
};

layout(local_size_x_id = 1) in;
)";


inline std::string temp_reduce_sum = R"(
#version 460
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable

layout(push_constant) uniform pushBlock {
        uint total;
};

layout(local_size_x = 32) in;

layout(binding=0) buffer buf_0 { float tensor_0[]; };
layout(binding=1) buffer buf_1 { float tensor_1[]; };
shared float data[64];


void main()
{
    float acc = 0;
    if(gl_GlobalInvocationID.x < total){
        acc = tensor_0[gl_GlobalInvocationID.x];
    }
    acc = subgroupAdd(acc);

    if(gl_SubgroupInvocationID == 0)
    {
        data[gl_SubgroupID] = acc;
    }

    memoryBarrierShared();
    barrier();

    if(gl_SubgroupID==0)
    {
        acc = gl_SubgroupInvocationID < gl_NumSubgroups ? data[gl_SubgroupInvocationID] : 0;
        acc = subgroupAdd(acc);
    }

    if(gl_LocalInvocationID.x == 0)
    {
        tensor_1[gl_WorkGroupID.x] = acc;
    }

}

)";

struct reduction_param
{
    uint32_t total;
};

inline uint32_t set_group_size(reduction_param p)
{
    return align_size(p.total, 32) / 32;
}

void reduce_sum(const tensor& t1, const tensor& t2)
{
    const reduction_param p { static_cast<uint32_t>(t1.get_size()) };
    // const std::string kernel_code = reduction_shader_code_math(reduction_shader, "subgroupAdd(acc)", t1, t2);
   
    k_runtime->make_job<reduction_param>("reduce_add", temp_reduce_sum, { t1.get_data(), t2.get_data()},
        p, set_group_size(p));
}


void reduce_mul(const tensor& t1, const tensor& t2)
{
    const reduction_param p { static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = reduction_shader_code_math(reduction_shader, "subgroupMul(acc)", t1, t2);
    k_runtime->make_job<reduction_param>("reduce_mul", kernel_code, { t1.get_data(), t2.get_data() },
        p, set_group_size(p));

}

void redcue_min(const tensor& t1, const tensor& t2)
{
    const reduction_param p { static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = reduction_shader_code_math(reduction_shader, "subgroupMin(acc)", t1, t2);
    k_runtime->make_job<reduction_param>("reduce_min", kernel_code, { t1.get_data(), t2.get_data() },
        p, set_group_size(p));

}

void reduce_max(const tensor& t1, const tensor& t2)
{
    const reduction_param p { static_cast<uint32_t>(t1.get_size()) };
    const std::string kernel_code = reduction_shader_code_math(reduction_shader, "subgroupMax(acc)", t1, t2);
    k_runtime->make_job<reduction_param>("reduce_max", kernel_code, { t1.get_data(), t2.get_data() },
        p, set_group_size(p));

}

void reduce_mean(const tensor& t1, const tensor& t2)
{
    
}

/*
 *
 *
#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

layout(std430, binding = 0) buffer Input
{
   float inputs[];
};

layout(std430, binding = 1) buffer Output
{
   float outputs[];
};

layout (local_size_x_id = 1) in;
layout (constant_id = 2) const int sumSubGroupSize = 64;

layout(push_constant) uniform PushConsts
{
  int n;
} consts;

shared float sdata[sumSubGroupSize];

void main()
{
    float sum = 0.0;
    if (gl_GlobalInvocationID.x < consts.n)
    {
        sum = inputs[gl_GlobalInvocationID.x];
    }

    sum = subgroupAdd(sum);

    if (gl_SubgroupInvocationID == 0)
    {
        sdata[gl_SubgroupID] = sum;
    }

    memoryBarrierShared();
    barrier();

    if (gl_SubgroupID == 0)
    {
        sum = gl_SubgroupInvocationID < gl_NumSubgroups ? sdata[gl_SubgroupInvocationID] : 0;
        sum = subgroupAdd(sum);
    }

    if (gl_LocalInvocationID.x == 0)
    {
        outputs[gl_WorkGroupID.x] = sum;
    }
}
 *
 *
 */