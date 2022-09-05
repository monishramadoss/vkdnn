// blas.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"
#define TILE_DIM 32

struct matmul_parameter
{
	uint32_t m;
	uint32_t n;
	uint32_t k;
};

inline uint32_t set_group_size_x(const matmul_parameter& p)
{
    return align_size(p.m, TILE_DIM) / TILE_DIM;
}

inline uint32_t set_group_size_y(const matmul_parameter& p)
{
    return align_size(p.n, TILE_DIM) / TILE_DIM;
}

std::string matmul_kernel_code = R"(
#define TILE_DIM 32
layout(push_constant) uniform pushBlock {
	uint m;
	uint n;
	uint k;
};

shared float ATile[TILE_DIM][TILE_DIM];
shared float BTile[TILE_DIM][TILE_DIM];

layout(local_size_x = TILE_DIM, local_size_y = TILE_DIM, local_size_z = 1) in;

)";

inline std::string matmul_shader(const std::string& kernel_shader_code, const tensor& t1, const tensor& t2, const tensor& t3)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[3];

    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, t1);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t2);
    const int ext_int_2 = tensor_injection(local_shader, shader_tensor[2], 2, t3);
    if (ext_int_0 == ext_int_1 && ext_int_0 == ext_int_2)
        local_shader = shader_extensions[ext_int_0] + local_shader;
    else if (ext_int_0 == ext_int_1 && ext_int_0 != ext_int_2)
    {
        local_shader = shader_extensions[ext_int_0] + local_shader;
        local_shader = shader_extensions[ext_int_2] + local_shader;
    }
    else if (ext_int_0 != ext_int_1 && ext_int_0 == ext_int_2)
    {
        local_shader = shader_extensions[ext_int_0] + local_shader;
        local_shader = shader_extensions[ext_int_1] + local_shader;
    }
    else if (ext_int_1 == ext_int_2 && ext_int_0 != ext_int_1)
    {
        local_shader = shader_extensions[ext_int_1] + local_shader;
        local_shader = shader_extensions[ext_int_2] + local_shader;
    }
    else
    {
        local_shader = shader_extensions[ext_int_0] + local_shader;
        local_shader = shader_extensions[ext_int_1] + local_shader;
        local_shader = shader_extensions[ext_int_2] + local_shader;
    }
    local_shader = "#version 460\n" + local_shader;

    local_shader += R"(
void main() {
    uint bx = gl_WorkGroupID.x; // workgroupsize
    uint by = gl_WorkGroupID.y;
    uint thrX = gl_LocalInvocationID.x;
    uint thrY = gl_LocalInvocationID.y;
    uint col = gl_GlobalInvocationID.x; // bx * gl_workGroupSize.x + thrX;
    uint row = gl_GlobalInvocationID.y; // by * gl_WorkGroupSize.y + thrY;

    float elementC = 0;

    for (int t = 0; t < (n-1)/TILE_DIM +1; ++t)
    {
        //threads to load matrix A to shared memory
        if(row < m && t*TILE_DIM+thrX < n)
            ATile[thrY][thrX] = tensor_0[row*n + t*TILE_DIM+thrX];
        else
            ATile[thrY][thrX] = 0.0f;

        //threads to load matrix B to shared memory
        if (t*TILE_DIM+thrY < n && col < k)
            BTile[thrY][thrX] = tensor_1[(t*TILE_DIM+thrY)*k + col];
        else
            BTile[thrY][thrX] = 0.0f;

        barrier();
        //calculate a partial value of thread element in C
        for (int i = 0; i < TILE_DIM; ++i)
            elementC += ATile[thrY][i] * BTile[i][thrX];
        barrier();
    }
    //copy final element value to the C matrix
    if (row < m && col < k)
        tensor_2[row*k+col] = elementC;

})";

    return local_shader;
}


inline void matmul(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const matmul_parameter p{ t1.get_shape(0), t1.get_shape(1), t2.get_shape(1) };
    auto& dev = k_runtime->get_device();
    const std::string kernel_code = matmul_shader(matmul_kernel_code, t1, t2, t3);
    dev.make_job<matmul_parameter>("matmul", kernel_code,{ t1.get_data(), t2.get_data() , t3.get_data()}, p, set_group_size_x(p), set_group_size_y(p));
}


inline void gemm(const tensor& t1, const tensor& t2, const tensor& t3, const float a, const float b, const tensor& t4)
{
    
}


inline void gemv(const tensor& t1, const tensor& t2, const float a,  const tensor& t3)
{
    
}


inline void saxy(const tensor& t1, const tensor& t2, const tensor& t3, const float a)
{
    
}