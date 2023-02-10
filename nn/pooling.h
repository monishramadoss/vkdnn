#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"


#define TILE_DIM 32

inline std::string pool_kernel_code = R"(
layout(push_constant) uniform pushBlock{
    uint batch_size;
    uint in_channel;
    uint input_size;
    uint output_size;
};

layout(local_size_x=TILE_DIM, local_size_y=1, local_size_z=1) in;
)";

struct pool_param {
    uint32_t batch_size;
    uint32_t in_channel;

    uint32_t input_size;
    uint32_t output_size;
};

inline uint32_t set_group_size_x(const pool_param& p)
{
    return align_size(p.output_size, TILE_DIM) / TILE_DIM;
}



template <uint32_t ndims>
void maxpoolNd(tensor& param, tensor& x, tensor& y){
    const pool_param p {x.get_shape(0), x.get_shape(1), (uint32_t)x.get_size(2), (uint32_t)y.get_size(2)};
    std::string kernel_code = "#define TILE_DIM " + std::to_string(TILE_DIM) + "\n#define NDIMS " + std::to_string(ndims) + "\n" + pool_kernel_code;
    kernel_code = inplace_pool_functions_kernel(kernel_code, "max(%s, %s);", param, x, y);
    k_runtime->make_job<pool_param>("maxPoolNd", kernel_code, {param.get_data(), x.get_data(), y.get_data()}, p, set_group_size(p));
}

void maxpool1d(tensor& param, tensor& x, tensor& y){
    maxpoolNd<1>(param, x, y);
}

void maxpool2d(tensor& param, tensor& x, tensor& y){
    maxpoolNd<2>(param, x, y);
}

void maxpool3d(tensor& param, tensor& x, tensor& y){
    maxpoolNd<3>(param, x, y);
}



template <uint32_t ndims>
void avgpoolNd(tensor& param, tensor& x, tensor& y){
    const pool_param p {x.get_shape(0), x.get_shape(1), (uint32_t)x.get_size(2), (uint32_t)y.get_size(2)};
    std::string kernel_code = "#define TILE_DIM " + std::to_string(TILE_DIM) + "\n#define NDIMS " + std::to_string(ndims) + "\n" + pool_kernel_code;
    kernel_code = inplace_pool_functions_kernel(kernel_code, " 1 / kernel_size * %s + %s;", param, x, y);
    k_runtime->make_job<pool_param>("maxPoolNd", kernel_code, {param.get_data(), x.get_data(), y.get_data()}, p, set_group_size(p));
}

void avgpool1d(tensor& param, tensor& x, tensor& y){
    avgpoolNd<1>(param, x, y);
}

void avgpool2d(tensor& param, tensor& x, tensor& y){
    avgpoolNd<2>(param, x, y);
}

void avgpool3d(tensor& param, tensor& x, tensor& y){
    maxpoolNd<3>(param, x, y);
}
