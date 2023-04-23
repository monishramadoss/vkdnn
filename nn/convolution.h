#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

#define TILE_DIM 32


static std::string conv_offset_engine_code = R"(
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

// TODO add batching into the offset engine using channel and batch to index the j dim

inline std::string inplace_unfold_functions_kernel(const std::string& kernel_shader_code, std::string fn_string, const tensor& pt, const tensor& t1, const tensor& t2, const tensor& t3)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[4];
    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, pt);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t1);
    const int ext_int_2 = tensor_injection(local_shader, shader_tensor[2], 2, t2);
    const int ext_int_3 = tensor_injection(local_shader, shader_tensor[3], 3, t3);

    if (ext_int_1 != ext_int_2 && ext_int_1 != ext_int_3)
        return " ERROR ";

    if (ext_int_0 != ext_int_1 && ext_int_0 != ext_int_2 && ext_int_0 != ext_int_3)
        local_shader = shader_extensions[ext_int_0] + local_shader;

    if (ext_int_1 == ext_int_2 && ext_int_1 == ext_int_3)
        local_shader = shader_extensions[ext_int_1] + local_shader;
    else if (ext_int_1 == ext_int_2 && ext_int_1 != ext_int_3)
    {
        local_shader = shader_extensions[ext_int_1] + local_shader;
        local_shader = shader_extensions[ext_int_3] + local_shader;
    }
    else if (ext_int_1 != ext_int_2 && ext_int_1 == ext_int_3)
    {
        local_shader = shader_extensions[ext_int_1] + local_shader;
        local_shader = shader_extensions[ext_int_2] + local_shader;
    }
    else if (ext_int_2 == ext_int_3 && ext_int_1 != ext_int_2)
    {
        local_shader = shader_extensions[ext_int_2] + local_shader;
        local_shader = shader_extensions[ext_int_3] + local_shader;
    }
    else
    {
        local_shader = shader_extensions[ext_int_1] + local_shader;
        local_shader = shader_extensions[ext_int_2] + local_shader;
        local_shader = shader_extensions[ext_int_3] + local_shader;
    }



    local_shader += "#define k(x) " + shader_tensor[0] + "[x]\n";
    local_shader += "#define s(x) " + shader_tensor[0] + "[x + NDIMS]\n";
    local_shader += "#define p(x) " + shader_tensor[0] + "[x + NDIMS*2]\n";
    local_shader += "#define d(x) " + shader_tensor[0] + "[x + NDIMS*3]\n";
    local_shader += "#define i(x) " + shader_tensor[0] + "[x + NDIMS*4]\n";
    local_shader += "#define o(x) " + shader_tensor[0] + "[x + NDIMS*5]\n";

    local_shader = SHADER_VERSION + local_shader;
    std::string tmp_type;
    gen_type(t3.get_type(), tmp_type);

    local_shader += R"(

shared )" + tmp_type + R"( ATile[TILE_DIM][TILE_DIM];
shared )" + tmp_type + R"( BTile[TILE_DIM][TILE_DIM];

)" + conv_offset_engine_code + R"(

void main() {

    uint bx = gl_WorkGroupID.x; 
    uint by = gl_WorkGroupID.y;
    
    uint thrX = gl_LocalInvocationID.x;
    uint thrY = gl_LocalInvocationID.y;

    uint col = gl_GlobalInvocationID.x; // bx * gl_workGroupSize.x + thrX;
    uint row = gl_GlobalInvocationID.y; // by * gl_WorkGroupSize.y + thrY;

    uint m = out_channel;
    uint k = output_size;
    uint n = in_channel * kernel_size;

    )" + tmp_type + R"( elementC = 0;
    for(uint t = 0; t < (n-1) / TILE_DIM + 1; ++t){
        if(row < m && t * TILE_DIM + thrX < n)
            ATile[thrY][thrX] = )" + shader_tensor[2] + R"([row*n + t*TILE_DIM+thrX];
        else
            ATile[thrY][thrX] = 0.0f;
        int in_offset = offset_engine(t*TILE_DIM+thrY, col);
        if (t*TILE_DIM+thrY < n && col < k && in_offset != -1)
            BTile[thrY][thrX] = )" + shader_tensor[1] + R"([in_offset];
        else
            BTile[thrY][thrX] = 0.0f;
        barrier();
        for(int tt = 0; tt < TILE_DIM; ++tt)
            elementC = )" + Format(fn_string, "ATile[thrY][tt]", "BTile[tt][thrX]", "elementC") + R"(
        barrier();
    }
    if(row < m && col < k)
        )" + shader_tensor[3] + R"([row * k + col] = elementC;
})";


    return local_shader;
}

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
    const conv_param p {x.get_shape(0), w.get_shape(0), w.get_shape(1), (uint32_t)x.get_size(2), (uint32_t)w.get_size(2), (uint32_t)y.get_size(2) };
    std::string kernel_code = "#define TILE_DIM " + std::to_string(TILE_DIM) + "\n#define NDIMS " + std::to_string(ndims) + "\n" + conv_kernel_code;
    kernel_code = inplace_unfold_functions_kernel(kernel_code, "fma(%s, %s, %s);", param, x, w, y);
    k_runtime->make_job<conv_param>("convNd", kernel_code, {param.get_data(), x.get_data(), w.get_data(), y.get_data()}, p,
        set_group_size_x(p), set_group_size_y(p), 1);
}

template<uint32_t ndims>
void deconvND(tensor& param, tensor& x, tensor& w, tensor& y) {
    const conv_param p{x.get_shape(0), w.get_shape(0), w.get_shape(1), (uint32_t)x.get_size(2), (uint32_t)w.get_size(2), (uint32_t)y.get_size(2)} ;
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
