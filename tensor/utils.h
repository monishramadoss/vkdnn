#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <string_view>

#define SHADER_VERSION "#version 460\n"
#define SUBGROUP_ENABLE "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"

template <typename... Args>
std::string Format(const std::string_view message, Args... formatItems)
{
    auto x = message.data();

    int size_s = std::snprintf(nullptr, 0, x, formatItems...) + 1; // Extra space for '\0'
    if (size_s <= 0)
    {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, x, formatItems...);

    return std::string(buf.get(), buf.get() + size - 1);
}

inline std::string singlton_shader_code(const std::string &kernel_shader_code, const std::string_view fn_pass, const tensor &t1)
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

inline std::string unary_shader_code(const std::string &kernel_shader_code, const std::string_view fn_pass, const tensor &t1, const tensor &t2)
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

inline std::string binary_shader_code(const std::string &kernel_shader_code, const std::string_view fn_pass, const tensor &t1, const tensor &t2, const tensor &t3)
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

inline std::string reduction_shader_code_math(const std::string &kernel_shader_code, const std::string fn_pass, const tensor &t1, const tensor &t2)
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

    local_shader = local_shader + "shared " + output_type + " sdata[32]; " + "\nvoid main()\n{\t\n\t" + output_type + " acc = 0; \n" +
                   "\tif(gl_GlobalInvocationID.x < total){\n\t\tacc = " + shader_tensor[0] +
                   "[gl_GlobalInvocationID.x];\n\t}\n\tacc = " + fn_pass +
                   ";\n\n\tif(gl_SubgroupInvocationID == 0)\n\t{\n\t\tsdata[gl_SubgroupID] = acc;\n\t}\n\n\tmemoryBarrierShared();\n\tbarrier();\n\n\tif(gl_SubgroupID==0)\n\t{\n\t\tacc = gl_SubgroupInvocationID < gl_NumSubgroups ? sdata[gl_SubgroupInvocationID] : 0;\n\t\tacc = " + fn_pass + ";\n\t}\n\n\tif(gl_LocalInvocationID.x == 0)\n\t{\n\t\t" + shader_tensor[1] +
                   "[gl_WorkGroupID.x] = acc;\n\t}\n}";

    return local_shader;
}

inline void parameter_shader_code(std::string shader_tensor[3], const std::string &kernel_shader_code, const tensor &pt, const tensor &t1, const tensor &t2)
{
    std::string local_shader = kernel_shader_code;
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

}

inline std::string transpose_kernel_code(const std::string &kernel_shader_code, const tensor &pt, const tensor &t1, const tensor &t2)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[3];
    parameter_shader_code(shader_tensor, local_shader, pt, t1, t2);

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
)",
                           shader_tensor[0].c_str(), shader_tensor[0].c_str(), shader_tensor[0].c_str(), shader_tensor[0].c_str(), shader_tensor[2].c_str(), shader_tensor[1].c_str());

    return local_shader;
}


inline std::string inplace_unfold_functions_kernel(const std::string &kernel_shader_code, std::string fn_string, const tensor &pt, const tensor &t1, const tensor &t2, const tensor &t3)
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

    local_shader +=  R"(

shared )" + tmp_type + R"( ATile[TILE_DIM][TILE_DIM];
shared )" + tmp_type + R"( BTile[TILE_DIM][TILE_DIM];

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
}


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
            ATile[thrY][thrX] =)" + shader_tensor[2] + R"([row*n + t*TILE_DIM+thrX];
        else
            ATile[thrY][thrX] = 0.0f;
        int in_offset = offset_engine(t*TILE_DIM+thrY, col);
        if (t*TILE_DIM+thrY < n && col < k && in_offset != -1)
            BTile[thrY][thrX] =)" + shader_tensor[1] + R"([in_offset];
        else
            BTile[thrY][thrX] = 0.0f;
        barrier();
        for(int tt = 0; tt < TILE_DIM; ++tt)
            elementC = )" + Format(fn_string, "ATile[thrY][tt]", "BTile[tt][thrX]", "elementC")  + R"(
        barrier();
    }
    if(row < m && col < k)
        )" + shader_tensor[3] + R"([row * k + col] = elementC;
})";


    return local_shader;
}


inline std::string inplace_pool_functions_kernel(const std::string &kernel_shader_code, std::string fn_string, const tensor &pt, const tensor &t1, const tensor &t2)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor[3];


    const int ext_int_0 = tensor_injection(local_shader, shader_tensor[0], 0, pt);
    const int ext_int_1 = tensor_injection(local_shader, shader_tensor[1], 1, t1);
    const int ext_int_2 = tensor_injection(local_shader, shader_tensor[2], 2, t2);    

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

    local_shader += "#define k(x) " + shader_tensor[0] + "[x]\n";
    local_shader += "#define s(x) " + shader_tensor[0] + "[x + NDIMS]\n";
    local_shader += "#define p(x) " + shader_tensor[0] + "[x + NDIMS*2]\n";
    local_shader += "#define d(x) " + shader_tensor[0] + "[x + NDIMS*3]\n";
    local_shader += "#define i(x) " + shader_tensor[0] + "[x + NDIMS*4]\n";
    local_shader += "#define o(x) " + shader_tensor[0] + "[x + NDIMS*5]\n";
    local_shader = SHADER_VERSION + local_shader;
    
    std::string tmp_type;
    gen_type(t1.get_type(), tmp_type);

    local_shader +=  R"(
shared )" + tmp_type + R"( ATile[TILE_DIM];

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
}

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
            elementC = )" + Format(fn_string, "ATile[tt]", "elementC")  + R"(
        barrier();
    }
    if(col < k)
        )" + shader_tensor[2] + R"([col] = elementC;


})";

    return local_shader;
}