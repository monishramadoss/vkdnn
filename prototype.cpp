// vulkan_dnn.cpp : Defines the entry point for the application.
//

#include "vulkan_dnn.h"

using namespace std;

struct param {
	int total;
};

std::string set_one_shader = R"(
#version 460
layout(push_constant) uniform pushBlock {
      int total;     
};

layout(binding = 0) readonly buffer buf1 { float X[]; };
layout(binding = 1) writeonly buffer buf2 {  float Y[]; };

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

void main()
{
    for (int i = int(gl_GlobalInvocationID.x); i < total; i += int(gl_NumWorkGroups.x * gl_WorkGroupSize.x))
    {
		Y[i] = X[i] + 2.0f;
    }
}

)";



int main()
{
	/*auto dev = kRuntime.get_device(0);
	vk_block* x = dev.malloc(128);
	vk_block* y = dev.malloc(128);
	memset(x->ptr, 0, 128);
	param p{ 128 / 4,};
	dev.kernel_mapping["set_one"] = set_one_shader;

	auto* j = dev.make_job("set_one", { x, y }, (void*)&p, sizeof(p));

	dev.run();

	float* f0 = static_cast<float*>(x->ptr);
	char* c1 = static_cast<char*>(y->ptr) + y->offset;
	
	float* f1 = reinterpret_cast<float*>(c1);
	for(auto i = 0; i < 128/4; ++i)
	{
		if (f1[i] != 2.0f)
			std::cout << i << '\n';
	}

	*/


	tensor t1 = tensor({ 2, 4, 4, 2 });
	tensor t2 = tensor({ 4, 8 });
	return 0;
}
