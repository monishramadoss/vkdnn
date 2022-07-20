#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"

struct param {
	int total;
};


std::string add_shader = R"(
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


void add(tensor& t1, tensor& t2, tensor& t3)
{
	param p;
	p.total = t1.get_size();
	auto dev = kRuntime.get_device();
	dev.kernel_mapping["add"] = add_shader;
	dev.make_job("add", { t1.get_data(), t2.get_data(), t3.get_data() }, &p, sizeof(param));

}

void sub(tensor& t1, tensor& t2, tensor& t3)
{

}

void mul(tensor& t1, tensor& t2, tensor& t3)
{

}

void true_div(tensor& t1, tensor& t2, tensor& t3)
{

}

void mod(tensor& t1, tensor& t2, tensor& t3)
{

}
#pragma once
