#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"
#include "base.h"

#include <cstdio>
#include <cstring>

struct binary_parameter{
	int total;
};

inline std::string binary_shader = R"(
#version 460
layout(push_constant) uniform pushBlock {
	int total;
};

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

)";




inline std::string add_shader = R"(
#version 460
layout(push_constant) uniform pushBlock {
      int total;     
};

layout(binding = 0) readonly buffer buf1 { float X[]; };
layout(binding = 0) readonly buffer buf1 { float W[]; };
layout(binding = 1) writeonly buffer buf2 {  float Y[]; };

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

void main()
{
    for (int i = int(gl_GlobalInvocationID.x); i < total; i += int(gl_NumWorkGroups.x * gl_WorkGroupSize.x))
    {
		Y[i] = X[i] + W[i];
    }
}

)";

inline void add(const tensor& t1, const tensor& t2, const tensor& t3)
{
	const binary_parameter p{ static_cast<int>(t1.get_size()) };
	auto dev = kRuntime.get_device();


	dev.kernel_mapping["add"] = binary_shader;
	std::string shader_tensor[3];


	int ext_int_0 = tensor_injection(dev.kernel_mapping["add"], shader_tensor[0], 0, t1);
	int ext_int_1 = tensor_injection(dev.kernel_mapping["add"], shader_tensor[1], 1, t2);
	int ext_int_2 = tensor_injection(dev.kernel_mapping["add"], shader_tensor[2], 2, t3);

	std::cout << dev.kernel_mapping["add"] << "\n";

//	dev.kernel_mapping["add"] += std::format(R"(
//int main (){
//	for (int i = int(gl_GlobalInvocationID.x); i < total; i += int(gl_NumWorkGroups.x * gl_WorkGroupSize.x))
//    {
//		{1}[i] = {0}[i] + {2}[i];
//    }
//})", shader_tensor[0], shader_tensor[1], shader_tensor[2]);


	dev.make_job<binary_parameter>("add", { t1.get_data(), t2.get_data(), t3.get_data() }, p);
}

inline void sub(const tensor& t1,const tensor& t2,const tensor& t3)
{
	binary_parameter p{ static_cast<int>(t1.get_size()) };
	auto dev = kRuntime.get_device();
	dev.kernel_mapping["sub"] = add_shader;
	dev.make_job<binary_parameter>("sub", { t1.get_data(), t2.get_data(), t3.get_data() }, p);
}

void mul(const tensor& t1,const tensor& t2,const tensor& t3)
{
	binary_parameter p{ static_cast<int>(t1.get_size()) };
	auto dev = kRuntime.get_device();
	dev.kernel_mapping["mul"] = add_shader;
	dev.make_job<binary_parameter>("mul", { t1.get_data(), t2.get_data(), t3.get_data() }, p);
}

inline void true_div(const tensor& t1,const tensor& t2,const tensor& t3)
{
	binary_parameter p{ static_cast<int>(t1.get_size()) };
	auto dev = kRuntime.get_device();
	dev.kernel_mapping["tdiv"] = add_shader;
	dev.make_job<binary_parameter>("tdiv", { t1.get_data(), t2.get_data(), t3.get_data() }, p);
}

inline void mod(const tensor& t1,const tensor& t2,const tensor& t3)
{
	binary_parameter p{ static_cast<int>(t1.get_size()) };
	auto dev = kRuntime.get_device();
	dev.kernel_mapping["mod"] = add_shader;
	dev.make_job<binary_parameter>("mod", { t1.get_data(), t2.get_data(), t3.get_data() }, p);
}
#pragma once
