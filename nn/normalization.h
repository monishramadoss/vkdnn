// normalization.h : Header file for your target.

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



inline std::string norm_shader_code(const std::string& kernel_shader_code, const tensor& t1, const tensor& t2, const tensor& t3, const tensor& t4, const tensor& t5, const tensor& t6) {

    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[6];
    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, t1);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[5], 1, t2);
    const int ext_int_2 = tensor_injection(local_shader, shader_tensor[2], 2, t3);
    const int ext_int_3 = tensor_injection(local_shader, shader_tensor[3], 3, t4);
    const int ext_int_4 = tensor_injection(local_shader, shader_tensor[4], 4, t5);
    const int ext_int_5 = tensor_injection(local_shader, shader_tensor[1], 5, t6);


    if (ext_int_0 == ext_int_1)
        local_shader = shader_extensions[ext_int_0] + local_shader;

    if (ext_int_1 == ext_int_2) {
        local_shader = shader_extensions[ext_int_1] + local_shader;
    }
    else
    {
        local_shader = shader_extensions[ext_int_0] + local_shader;
        local_shader = shader_extensions[ext_int_1] + local_shader;
        local_shader = shader_extensions[ext_int_2] + local_shader;
    }

    local_shader = SHADER_VERSION + local_shader;
    std::string tmp_type;
    gen_type(t1.get_type(), tmp_type);

    local_shader += R"(

shared )" + tmp_type + R"( thread_mean[THREAD_COUNT];
shared )" + tmp_type + R"( thread_m2[THREAD_COUNT];
shared uint thread_count[THREAD_COUNT];


void welford_combine(float val, inout uint count, inout float mean, inout float m2)
{
    count += 1;
    float delta1 = val - mean;
    mean = delta1 / count;
    float delta2 = val - mean;
    m2 += delta1 * delta2;
}

void welford_combine_2(uint count_b, )" + tmp_type + R"(  mean_b, )" + tmp_type + R"(  m2_b, inout uint count_a, inout )" + tmp_type + R"(  mean_a, inout )" + tmp_type + R"(  m2_a)
{
    uint count = count_b + count_a;
    )" + tmp_type + R"( nb = count_b / count;
    )" + tmp_type + R"( delta = mean_b - mean_a;
    mean_a += delta * nb;
    m2_a += m2_b + delta * count_a * nb;
    count_a = count;
} 

void welford_reduce(uint tidx){
    welford_combine_2(thread_count[tidx + 64], thread_mean[tidx + 64], thread_m2[tidx + 64],
        thread_count[tidx], thread_mean[tidx], thread_m2[tidx]
    );
    memoryBarrierShared();
    welford_combine_2(thread_count[tidx + 32], thread_mean[tidx + 32], thread_m2[tidx + 32],
        thread_count[tidx], thread_mean[tidx], thread_m2[tidx]
    );
        memoryBarrierShared();
    welford_combine_2(thread_count[tidx + 16], thread_mean[tidx + 16], thread_m2[tidx + 16],
        thread_count[tidx], thread_mean[tidx], thread_m2[tidx]
    );
    memoryBarrierShared();
    welford_combine_2(thread_count[tidx + 8], thread_mean[tidx + 8], thread_m2[tidx + 8],
        thread_count[tidx], thread_mean[tidx], thread_m2[tidx]
    );
    memoryBarrierShared();
    welford_combine_2(thread_count[tidx + 4], thread_mean[tidx + 4], thread_m2[tidx + 4],
        thread_count[tidx], thread_mean[tidx], thread_m2[tidx]
    );
    memoryBarrierShared();
    welford_combine_2(thread_count[tidx + 2], thread_mean[tidx + 2], thread_m2[tidx + 2],
        thread_count[tidx], thread_mean[tidx], thread_m2[tidx]
    );    
    memoryBarrierShared();
    welford_combine_2(thread_count[tidx + 1], thread_mean[tidx + 1], thread_m2[tidx + 1],
        thread_count[tidx], thread_mean[tidx], thread_m2[tidx]
    );
    memoryBarrierShared();
}

void main(){
    
    uint rows = batch_size;
    uint cols = input_size * in_channel;
    
    uint tid = gl_LocalInvocationID.x;
    uint nwg = gl_NumWorkGroups.x;
    uint wgid = gl_WorkGroupID.x;
    uint blkS = gl_WorkGroupSize.x;


    for(uint row = wgid; row < rows; row += nwg) {  

        thread_count[tid] = 0;
        thread_mean[tid] =  )" + tmp_type + R"((0);
        thread_m2[tid] =  )" + tmp_type + R"((0);

        for(uint col = tid; col < cols; col += blkS)
            welford_combine()" + shader_tensor[0] + R"([row * cols + col], thread_count[tid], thread_mean[tid], thread_m2[tid]);

        welford_reduce(tid);

        )" + tmp_type + R"( row_variance = max(thread_m2[0] / thread_count[0], 0);
        )" + tmp_type + R"( row_inv_var = inversesqrt(row_variance + epsilon);
        )" + tmp_type + R"( row_mean = thread_mean[0];

        if(tid == 0){
            )" + shader_tensor[1] + R"([row] = momentum * row_mean + (1 - momentum) * )" + shader_tensor[1] + R"([row];
            )" + shader_tensor[2] + R"([row] = momentum * row_variance + (1 - momentum) * )" + shader_tensor[2] + R"([row];
        }


        for(uint col = tid; col < cols; col += blkS)
            )" + shader_tensor[5] + R"([row * cols + col] = ()" + shader_tensor[0] + R"([row * cols + col] - row_mean) * row_inv_var * )" + shader_tensor[3] + R"([col] + )" + shader_tensor[4] + R"([col];              
      
    }
   
})";
    return local_shader;
}


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
    const norm_param p {x.get_shape(0), x.get_shape(1), (uint32_t)x.get_size(2), 0.1f, 0.00001f};
    uint32_t rows = p.batch_size;
    uint32_t cols = p.input_size * p.in_channel;
    norm_kernel_code = "#define THREAD_COUNT " + std::to_string(THREAD_COUNT) + "\n" + norm_kernel_code;
    std::string kernel_code = norm_shader_code(norm_kernel_code, x, m, v, w,b, y);
    
    k_runtime->make_job<norm_param>("batchnorm", kernel_code, {
        x.get_data(), m.get_data(), v.get_data(), w.get_data(), b.get_data(), y.get_data(),
    }, p, set_group_size_x(p));   
}

void instancenorm(const tensor &x, const tensor &m, const tensor &v, const tensor &w, const tensor &b, const tensor &y){
    const norm_param p {x.get_shape(0), x.get_shape(1), (uint32_t)x.get_size(2), 0.1f, 0.00001f};
    uint32_t rows = p.batch_size;
    uint32_t cols = p.input_size * p.in_channel;
    norm_kernel_code = "#define THREAD_COUNT " + std::to_string(THREAD_COUNT) + "\n" + norm_kernel_code;
    std::string kernel_code = norm_shader_code(norm_kernel_code, x, m, v, w, b, y);
    
    k_runtime->make_job<norm_param>("instancenorm", kernel_code, {
        x.get_data(), m.get_data(), v.get_data(), w.get_data(), b.get_data(), y.get_data(),
    }, p, set_group_size_x(p));   
}

void layernorm(const tensor &x, const tensor &m, const tensor &v, const tensor &w, const tensor &b, const tensor &y){
    const norm_param p {x.get_shape(0) * x.get_shape(1), 1, (uint32_t)x.get_size(2), 0.1f, 0.00001f};
    uint32_t rows = p.batch_size;
    uint32_t cols = p.input_size * p.in_channel;
    norm_kernel_code = "#define THREAD_COUNT " + std::to_string(THREAD_COUNT) + "\n" + norm_kernel_code;
    std::string kernel_code = norm_shader_code(norm_kernel_code, x, m, v, w, b, y);
    
    k_runtime->make_job<norm_param>("layernorm", kernel_code, {
        x.get_data(), m.get_data(), v.get_data(), w.get_data(), b.get_data(), y.get_data(),
    }, p, set_group_size_x(p));   
}



