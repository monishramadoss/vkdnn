#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <string_view>

#define SHADER_VERSION "#version 460\n"
#define SUBGROUP_ENABLE "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"


template<typename... Args>
std::string Format(const std::string_view message, Args... formatItems)
{
    auto x = message.data();

    int size_s = std::snprintf( nullptr, 0, x, formatItems ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, x, formatItems ... );
    
    return  std::string(buf.get(), buf.get() + size - 1);
}


inline std::string singlton_shader_code(const std::string& kernel_shader_code, const std::string_view fn_pass, const tensor& t1)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor;
    const int ext_int_0 = tensor_injection(local_shader, shader_tensor, 0, t1);
    local_shader = shader_extensions[ext_int_0] + local_shader;

    local_shader = SHADER_VERSION + local_shader;

    local_shader += "\nvoid main() {\n\tfor (uint i = gl_GlobalInvocationID.x; i < total; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x){\n\t\t";
    local_shader += Format(fn_pass, shader_tensor.c_str()) + "\n\t}\n}";
    return local_shader;
}

inline std::string unary_shader_code(const std::string& kernel_shader_code, const std::string_view fn_pass, const tensor& t1, const tensor& t2)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[2];
    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, t1);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t2);

    if (ext_int_0 == ext_int_1)
        local_shader = shader_extensions[ext_int_0] + local_shader;
    else
    {
        local_shader = shader_extensions[ext_int_0] + local_shader;
        local_shader = shader_extensions[ext_int_1] + local_shader;
    }
    local_shader = SHADER_VERSION + local_shader;

    local_shader += "\nvoid main() {\n\tfor (uint i = gl_GlobalInvocationID.x; i < total; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x){\n\t\t";
    local_shader += Format(fn_pass, shader_tensor[1].c_str(), shader_tensor[0].c_str()) + "\n\t}\n}";
    return local_shader;
}

inline std::string binary_shader_code(const std::string& kernel_shader_code, const std::string_view fn_pass, const tensor& t1, const tensor& t2, const tensor& t3)
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
    local_shader = SHADER_VERSION + local_shader;

    local_shader += "\nvoid main() {\n\tfor (uint i = gl_GlobalInvocationID.x; i < total; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x){\n\t\t";
    local_shader += Format(fn_pass, shader_tensor[2].c_str(), shader_tensor[0].c_str(), shader_tensor[1].c_str()) + "\n\t}\n}";
    return local_shader;
}

inline std::string reduction_shader_code_math(const std::string& kernel_shader_code, const std::string fn_pass, const tensor& t1, const tensor& t2)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[2];
    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, t1);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t2);

    if (ext_int_0 == ext_int_1)
        local_shader = shader_extensions[ext_int_0] + local_shader;
    else
    {
        local_shader = shader_extensions[ext_int_0] + local_shader;
        local_shader = shader_extensions[ext_int_1] + local_shader;
    }
    local_shader = SUBGROUP_ENABLE + local_shader;
    local_shader = SHADER_VERSION + local_shader;

    std::string output_type;
    gen_type(t2.get_type(), output_type);

    local_shader = local_shader + "shared " +  output_type + " sdata[32]; " + "\nvoid main()\n{\t\n\t" + output_type + " acc = 0; \n" +
        "\tif(gl_GlobalInvocationID.x < total){\n\t\tacc = " + shader_tensor[0] +
        "[gl_GlobalInvocationID.x];\n\t}\n\tacc = " + fn_pass +
        ";\n\n\tif(gl_SubgroupInvocationID == 0)\n\t{\n\t\tsdata[gl_SubgroupID] = acc;\n\t}\n\n\tmemoryBarrierShared();\n\tbarrier();\n\n\tif(gl_SubgroupID==0)\n\t{\n\t\tacc = gl_SubgroupInvocationID < gl_NumSubgroups ? sdata[gl_SubgroupInvocationID] : 0;\n\t\tacc = "
        + fn_pass + ";\n\t}\n\n\tif(gl_LocalInvocationID.x == 0)\n\t{\n\t\t" + shader_tensor[1] +
        "[gl_WorkGroupID.x] = acc;\n\t}\n}";



    return local_shader;
}

inline std::string* parameter_shader_code(std::string& local_shader, const tensor& pt, const tensor& t1, const tensor& t2)
{
    std::string shader_tensor[3];
    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, pt);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t1);
    const int ext_int_2 = tensor_injection(local_shader, shader_tensor[2], 2, t2);

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
    local_shader = SHADER_VERSION + local_shader;

    return shader_tensor;
}


inline std::string transpose_kernel_code(const std::string& kernel_shader_code, const tensor& pt, const tensor& t1, const tensor& t2)
{
    std::string local_shader = kernel_shader_code;
    const std::string* shader_tensor = parameter_shader_code(local_shader, pt, t1, t2);

local_shader += Format(R"(
void main() {
    for(uint i = gl_GlobalInvocationID.x; i < total; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x){
        uint old_pos = 0;
        uint new_pos = i;
        for(uint j = 0; j < ndims; ++j){
            uint order = %s[j];
            old_pos += (new_pos / %s[ndims+j]) * %s[ndims*2 + order];
            new_pos %= %s[ndims+j];
        }
        %s[i] = %s[abs(old_pos)];
    }
}
)", shader_tensor[0].c_str(), shader_tensor[0].c_str(), shader_tensor[0].c_str(), shader_tensor[0].c_str(), shader_tensor[2].c_str(), shader_tensor[1].c_str());

    return local_shader;
}



inline std::string unfold_kernel_code(const std::string& kernel_shader_code, const tensor& pt, const tensor& t1, const tensor& t2)
{
    std::string local_shader = kernel_shader_code;
    const std::string* shader_tensor = parameter_shader_code(local_shader, pt, t1, t2);

    //return local_shader;

    local_shader += R"(

#define ksize(x) tensor_2[x]
#define stride(x) tensor_2[ndims + x]
#define padding(x) tensor_2[ndims*2 + x]
#define dilation(x) tensor_2[ndims*3 + x]
#define input(x) tensor_2[ndims*4 + x]
#define output(x) tensor_2[ndims*5 + x]

uint offsetEngine(uint offset_a, uint offset_b){
    uint i1 = offset_b % output(2*ndims - 1);
    uint i2 = offset_a % output(ndims-1);
    
    uint o1 = uint(offset_b / output(2*ndims - 1));
    uint o2 = uint(offset_a / output(ndims-1));
    uint o3 = uint(offset_a / ksize(ndims-1));
       

    uint o = i1 * stride(ndims-1) + i2 * dilation(ndims-1) - padding(ndims-1);
    
    for(uint k = ndims-1; k > 0; --k){
        i1 = o1 % output(ndims + k - 1);
        i2 = o2 % output(k - 1);
        
        o1 = uint(o1 / output(ndims + k - 1));
        o2 = uint(o2 / output(k - 1));
        o3 = uint(o3 / ksize(k - 1));
        o += (i1 * stride(k - 1) + i2 * dilation(k - 1) - padding(k - 1)) * input(k - 1);

    }
    return o + o3 * channel_stride;
 }


void main() {
   
    uint out_offset = 0;
    uint in_offset = 0;
    uint offset = 0;
    uint ob = output_size * batch_size;
    uint kc = kernel_size * channel_size;


    for(uint i = gl_GlobalInvocationID.x; i < ob; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x){
        for(uint j = gl_GlobalInvocationID.y; j < kc; j += gl_NumWorkGroups.y * gl_WorkGroupSize.y){
            in_offset  = offsetEngine(j, i);           
            out_offset = j * ob + i;
            tensor_1[out_offset] = tensor_0[in_offset];
        }
    }
}


)";

    return local_shader;
}