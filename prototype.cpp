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


	tensor r1 = arange_t<float>(1024);
	tensor r2 = zeros<float>({ 1024 });

	reduce_sum(r1, r2);
	r2.sync(false);

	
	tensor t1 = arange_t<float>(64);
	t1.reshape({ 8, 8 });
	tensor t2 = arange_t<float>(64);
	t2.reshape({ 8, 8 });
	tensor t3({ 8, 8 });
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
	const auto* f2 = static_cast<float*>(t6.get_host_data()->ptr);
	const auto* f3 = static_cast<float*>(r2.get_host_data()->ptr);
	





	k_runtime->wait();

	size_t i;
	for(i = 0; i < t6.get_size(); ++i)
	{
		if (f2[i] != 4096)
 			break;
	}

	bool val = i  == t6.get_size();
	std::cout << val << '\n';

	for (i = 0; i < 8; ++i)
	{
		for(size_t j = 0; j < 8; ++j)
		{
			std::cout << f1[i * 8 + j] << " ";
		}
		std::cout << '\n';
	}

	for(uint32_t k = 0; k < 1024; ++k)
	{
		if(f3[k] != 0)
			std::cout << k << " " << f3[k] << '\n';	
	}


	//CUDA :	60-80 microseconds;
	//VK :		1268 microseconds


	return 0;
}
