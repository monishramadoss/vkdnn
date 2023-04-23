// pooling.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

#define TILE_DIM 32



static std::string unfold_offset_engine_code = R"(
int offset_engine(uint offset_a, uint offset_b)
{   
    uint idx = NDIMS - 1 + 2;
    uint idx2 = idx - 2;
    uint index_a = offset_b % o(NDIMS + idx);
    uint index_b = offset_a % o(idx);
    uint o1 = offset_b / o(NDIMS + idx);
    uint o2 = offset_a / o(idx);
    uint o3 = offset_a / k(idx2);
    
    uint ot = index_a * s(idx2) + index_b * d(idx2) - p(idx2); 

    for(idx = idx-1; idx > 1; --idx){
        idx2 = idx - 2;
        index_a = o1 % o(NDIMS + idx);
        index_b = o2 % o(idx);
        uint offset = index_a * s(idx2) + index_b * d(idx2) - p(idx2);
        ot += offset * i(idx);
        if(offset >= i(idx))
            return -1;
        o1 = o1 / o(NDIMS + idx);
        o2 = o2 / o(idx);
        o3 = o3 / k(idx2);
    }

    return int(ot + o3 * input_size);
})";

inline std::string inplace_pool_functions_kernel(const std::string& kernel_shader_code, std::string fn_string, const tensor& pt, const tensor& t1, const tensor& t2)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[3];


    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, pt);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t1);
    const int ext_int_2 = tensor_injection(local_shader, shader_tensor[2], 2, t2);

    if (ext_int_0 == ext_int_1)
        local_shader = shader_extensions[ext_int_0] + local_shader;
    else
    {
        local_shader = shader_extensions[ext_int_0] + local_shader;
        local_shader = shader_extensions[ext_int_1] + local_shader;
    }

    local_shader += "#define k(x) " + shader_tensor[0] + "[x]\n";
    local_shader += "#define s(x) " + shader_tensor[0] + "[x + NDIMS]\n";
    local_shader += "#define p(x) " + shader_tensor[0] + "[x + NDIMS*2]\n";
    local_shader += "#define d(x) " + shader_tensor[0] + "[x + NDIMS*3]\n";
    local_shader += "#define i(x) " + shader_tensor[0] + "[x + NDIMS*4]\n";
    local_shader += "#define o(x) " + shader_tensor[0] + "[x + NDIMS*5]\n";
    local_shader = SHADER_VERSION + local_shader;

    std::string tmp_type;
    gen_type(t1.get_type(), tmp_type);

    local_shader += R"(
shared )" + tmp_type + R"( ATile[TILE_DIM];

)" + unfold_offset_engine_code + R"(


void main(){
    uint kernel_size = 1;
    for(uint i = 0; i < NDIMS; ++i)
        kernel_size *= k(i);

    uint bx = gl_WorkGroupID.x; 
    
    uint thrX = gl_LocalInvocationID.x;

    uint col = gl_GlobalInvocationID.x; // bx * gl_workGroupSize.x + thrX;
    
    uint k = output_size;
    uint n = in_channel * kernel_size;

    )" + tmp_type + R"( elementC = 0;
    for(uint t = 0; t < (n-1) / TILE_DIM + 1; ++t){
        int in_offset = offset_engine(t*TILE_DIM+thrX, col);
        if (t*TILE_DIM+thrX < n && col < k && in_offset != -1)
            ATile[thrX] = )" + shader_tensor[1] + R"([in_offset];
        else
            ATile[thrX] = 0.0f;
        barrier();
        for(int tt = 0; tt < TILE_DIM; ++tt)
            elementC = )" + Format(fn_string, "ATile[tt]", "elementC") + R"(
        barrier();
    }
    if(col < k)
        )" + shader_tensor[2] + R"([col] = elementC;


})";

    return local_shader;
}

inline std::string pool_kernel_code = R"(
layout(push_constant) uniform pushBlock {
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
    k_runtime->make_job<pool_param>("maxPoolNd", kernel_code, {param.get_data(), x.get_data(), y.get_data()}, p, set_group_size_x(p));
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
    k_runtime->make_job<pool_param>("maxPoolNd", kernel_code, {param.get_data(), x.get_data(), y.get_data()}, p, set_group_size_x(p));
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
