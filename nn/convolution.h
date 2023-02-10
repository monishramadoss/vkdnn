#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

#define TILE_DIM 32

inline std::string conv_kernel_code = R"(
layout(push_constant) uniform pushBlock{
    uint batch_size;
    uint in_channel;
    uint out_channel;
    uint input_size;
    uint kernel_size;
    uint output_size;
};


layout(local_size_x=TILE_DIM, local_size_y=TILE_DIM, local_size_z=1) in;
)";

struct conv_param
{
    uint32_t batch_size;
    uint32_t in_channel;
    uint32_t out_channel;

    uint32_t input_size;
    uint32_t kernel_size;
    uint32_t output_size;
};

inline uint32_t set_group_size_x(const conv_param& p)
{
    return align_size(p.batch_size * p.output_size, TILE_DIM) / TILE_DIM;
}

inline uint32_t set_group_size_y(const conv_param& p)
{
    return align_size(p.out_channel, TILE_DIM) / TILE_DIM;
}


template <uint32_t ndims>
void convND(tensor& param, tensor& x, tensor& w, tensor& y) { // todo add batching stuff to run multiple kernels
    const conv_param p {x.get_shape(0), w.get_shape(0), w.get_shape(1), (uint32_t)x.get_size(2), (uint32_t)w.get_size(2), (uint32_t)y.get_size() };
    std::string kernel_code = "#define TILE_DIM " + std::to_string(TILE_DIM) + "\n#define NDIMS " + std::to_string(ndims) + "\n" + conv_kernel_code;
    kernel_code = inplace_unfold_functions_kernel(kernel_code, "fma(%s, %s, %s);", param, x, w, y);
    k_runtime->make_job<conv_param>("convNd", kernel_code, {param.get_data(), x.get_data(), w.get_data(), y.get_data()}, p,
        set_group_size_x(p), set_group_size_y(p), 1);
}

template<uint32_t ndims>
void deconvND(tensor& param, tensor& x, tensor& w, tensor& y) {
    const conv_param p{x.get_shape(0), w.get_shape(0), w.get_shape(1), (uint32_t)x.get_size(2), (uint32_t)w.get_size(2), (uint32_t)y.get_size()} ;
    std::string kernel_code = "#define TILE_DIM " + std::to_string(TILE_DIM) + "\n#define NDIMS " + std::to_string(ndims) + "\n" + conv_kernel_code;
    kernel_code = inplace_unfold_functions_kernel(kernel_code, "%s / %s + %s;", param, x, w, y);
    k_runtime->make_job<conv_param>("convNd", kernel_code, {param.get_data(), x.get_data(), w.get_data(), y.get_data()}, p,
        set_group_size_x(p), set_group_size_y(p), 1);
}


void conv1d(tensor& param, tensor& x, tensor& w, tensor& y) {
    convND<1>(param, x, w, y);
}

void deconv1d(tensor& param, tensor& x, tensor& w, tensor& y) {
    deconvND<1>(param, x, w, y);
}

void conv2d(tensor& param, tensor& x, tensor& w, tensor& y) {
    convND<2>(param, x, w, y);
}

void deconv2d(tensor& param, tensor& x, tensor& w, tensor& y) {
    deconvND<2>(param, x, w, y);
}

void conv3d(tensor& param, tensor& x, tensor& w, tensor& y) {
    convND<3>(param, x, w, y);
}

void deconv3d(tensor& param, tensor& x, tensor& w, tensor& y) {
    deconvND<3>(param, x, w, y);
}
