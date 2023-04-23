#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <string_view>

#define SHADER_VERSION "#version 450\n"
#define SUBGROUP_ENABLE "#extension GL_KHR_shader_subgroup_arithmetic : enable\n"

template <typename... Args>
std::string format(const std::string_view message, Args... formatItems)
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



template<typename... arg_type>
std::string Format(const std::string_view& main, arg_type... args) {
    const int size = sizeof...(args);
    std::string s = std::string(main);
    std::string arg_list[size] = { args... };

    for (int i = 0; i < size; ++i) {
        std::string sub_str = "{" + std::to_string(i) + "}";
        std::string rep_str = arg_list[i];
        size_t pos = 0;
        while (pos += arg_list[i].length()) {
            pos = s.find(sub_str);
            if (pos == std::string::npos)
                break;
            s.erase(pos, sub_str.length());
            s.insert(pos, rep_str);
        }
    }
    return s;
}

inline std::string singlton_shader_code(const std::string &kernel_shader_code, const std::string_view fn_pass, const tensor &t1)
{
    std::string local_shader = kernel_shader_code;
    std::string shader_tensor;
    const int ext_int_0 = tensor_injection(local_shader, shader_tensor, 0, t1);
    local_shader = shader_extensions[ext_int_0] + local_shader;
    local_shader = SHADER_VERSION + local_shader;

    local_shader += R"(
void main() {
    for (uint i = gl_GlobalInvocationID.x; i < total; i += gl_NumWorkGroups.x * gl_WorkGroupSize.x) {
        )" + Format(fn_pass, shader_tensor.c_str()) + R"(
    }
}
    )";
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
    local_shader += Format(fn_pass, shader_tensor[0].c_str(), shader_tensor[1].c_str()) + "\n\t}\n}";
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
    local_shader += Format(fn_pass, shader_tensor[0].c_str(), shader_tensor[1].c_str(), shader_tensor[2].c_str()) + "\n\t}\n}";
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



