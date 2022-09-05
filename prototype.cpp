// vulkan_dnn.cpp : Defines the entry point for the application.
//
#include "vulkan_dnn.h"
#include <chrono>
using namespace std::chrono;

using namespace std;
struct temp_param
{
	int size;
};

int main()
{
	init();

	tensor t1 = arange_t<float>(25);
	t1.reshape({ 5, 5 });
	tensor t2 = arange_t<float>(25);
	t2.reshape({ 5, 5 });
	tensor t3({ 5, 5 });

	matmul(t1, t2, t3);

	t3.sync(false);
	


	// 16384 
	uint32_t m = 16384;
	uint32_t n = 8192;
	uint32_t k = 8192; 

	m = 4096;
	n = 4096;
	k = 4096;

	tensor t4 = tensor::ones<float>({ m, n });
	tensor t5 = tensor::ones<float>({ n, k });
	tensor t6 = tensor::zeros<float>({ m, k });

	matmul(t4, t5, t6);
	t6.sync(false);

	
	const auto* f1 = static_cast<float*>(t3.get_host_data()->ptr);
	//const auto* f2 = static_cast<float*>(t5.get_host_data()->ptr);
	const auto* f3 = static_cast<float*>(t6.get_host_data()->ptr);

	k_runtime->wait();

	size_t i;
	for(i = 0; i < t6.get_size(); ++i)
	{
		if (f3[i] != 4096)
 			break;
	}

	for (i = 0; i < 5; ++i)
	{
		for(size_t j = 0; j < 5; ++j)
		{
			std::cout << f1[i * 5 + j] << " ";
		}
		std::cout << '\n';
	}


	//CUDA : 60-80 microseconds;
	//VK : 

	return 0;
}
