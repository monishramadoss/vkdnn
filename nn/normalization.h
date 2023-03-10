#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"


#define THREAD_COUNT 128

inline std::string norm_kernel_code = R"(

layout(push_constant) uniform pushBlock {
    uint batch_size;
    uint in_channel;        
    uint input_size;	
    float momentum;
    float epsilon;
};

layout(local_size_x = THREAD_COUNT) in; 

)";


struct norm_param {
    uint32_t batch_size;
    uint32_t in_channel;
    uint32_t input_size;
    float momentum;
    float epsilon;
};

inline uint32_t set_group_size_x(const norm_param& p)
{
    return align_size(p.batch_size, 32) / 32;
}


//rows = batch_size;
//cols = input_size * in_channel;
// v, m ; rows
// w, b ; cols 

void batchnorm(const tensor &x, const tensor &m, const tensor &v, const tensor &w, const tensor &b, const tensor &y){
    const norm_param p {x.get_shape(0), x.get_shape(1), (uint32_t)x.get_size(2), 0.1, 0.00001};
    uint32_t rows = p.batch_size;
    uint32_t cols = p.input_size * p.in_channel;
    norm_kernel_code = "#define THREAD_COUNT " + std::to_string(THREAD_COUNT) + "\n" + norm_kernel_code;
    std::string kernel_code = norm_shader_code(norm_kernel_code, x, m, v, w,b, y);
    
    k_runtime->make_job<norm_param>("batchnorm", kernel_code, {
        x.get_data(), m.get_data(), v.get_data(), w.get_data(), b.get_data(), y.get_data(),
    }, p, set_group_size_x(p));   
}

void instancenorm(const tensor &x, const tensor &m, const tensor &v, const tensor &w, const tensor &b, const tensor &y){
    const norm_param p {x.get_shape(0), x.get_shape(1), (uint32_t)x.get_size(2), 0.1, 0.00001};
    uint32_t rows = p.batch_size;
    uint32_t cols = p.input_size * p.in_channel;
    norm_kernel_code = "#define THREAD_COUNT " + std::to_string(THREAD_COUNT) + "\n" + norm_kernel_code;
    std::string kernel_code = norm_shader_code(norm_kernel_code, x, m, v, w, b, y);
    
    k_runtime->make_job<norm_param>("instancenorm", kernel_code, {
        x.get_data(), m.get_data(), v.get_data(), w.get_data(), b.get_data(), y.get_data(),
    }, p, set_group_size_x(p));   
}

void layernorm(const tensor &x, const tensor &m, const tensor &v, const tensor &w, const tensor &b, const tensor &y){
    const norm_param p {x.get_shape(0) * x.get_shape(1), 1, (uint32_t)x.get_size(2), 0.1, 0.00001};
    uint32_t rows = p.batch_size;
    uint32_t cols = p.input_size * p.in_channel;
    norm_kernel_code = "#define THREAD_COUNT " + std::to_string(THREAD_COUNT) + "\n" + norm_kernel_code;
    std::string kernel_code = norm_shader_code(norm_kernel_code, x, m, v, w, b, y);
    
    k_runtime->make_job<norm_param>("layernorm", kernel_code, {
        x.get_data(), m.get_data(), v.get_data(), w.get_data(), b.get_data(), y.get_data(),
    }, p, set_group_size_x(p));   
}



