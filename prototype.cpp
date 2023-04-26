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

    // 16384 
    uint32_t m = 16384;
    uint32_t n = 8192;
    uint32_t k = 8192; 
    m = 4096;
    n = 4096;
    k = 4096;

    m = 512;
    n = 512;
    k = 512;

    float K = k;
    tensor t4 = tensor::ones<float>({ m, k });
    tensor t5 = tensor::ones<float>({ k, n });
    tensor t6 = tensor::zeros<float>({ m, n });

    gemm(t4, t5, t6, 1.0, 0.0);

    t4.sync(false);
    t5.sync(false);
    t6.sync(false);
    // TODO needs to update sync model because it is not working....


    const float* f1 = (float*)t4.get_host_data()->ptr;
    const float* f2 = (float*)t5.get_host_data()->ptr;
    const float* f3 = (float*)t6.get_host_data()->ptr;

    std::vector<uint32_t> idxs;
    std::vector<float> vals;
    std::vector<float> vals2;
    std::vector<float> vals3;
    size_t i;
    for (i = 0; i < t4.get_size(); ++i) {
        vals2.push_back(f1[i]);
    }
    for (i = 0; i < t5.get_size(); ++i) {
        vals3.push_back(f2[i]);
    }
    
    for(i = 0; i < t6.get_size(); ++i)
    {
        vals.push_back(f3[i]);
        if(abs(f3[i] - k) > 0.5){

            idxs.push_back(i);
            
        }
    }
    size_t tmp = t6.get_size();
    std::cout << f3[0] << " " << f3[tmp-1] << std::endl;
    std::cout << idxs.size() << std::endl;
    bool val = idxs.size()  == 0;


    std::cout << val << std::endl;


    /*float acc = 0;
    for(uint32_t k = 0; k < 2048; ++k)
    {
        if(f3[k] != 0)
            std::cout << k << " " << f3[k] << '\n';
        acc += f3[k];
    }
    std::cout << acc << '\n';
    */

    //CUDA :	60-80 microseconds;
    //VK :		1268 microseconds

    // tensor inpt = arange_t(0.0f, 18.0f, 1.0f);
    // inpt.reshape({ 1, 2, 3, 3 });	
    // tensor wght = tensor::ones<float>({ 2,4,2,2 });




    // std::vector<uint32_t> params = {
    // 	2, 2,		// kernel_size
    // 	1, 1,		// stride
    // 	0, 0,		// padding
    // 	1, 1,		// dilation	
    // 	1, 2, 3, 3,	// input shape
    // 	2, 2, 2, 2		// output shape
    // };
    // tensor oupt = tensor::zeros<float>({1, 4, 4});
    // auto param_tensor = tensor({ static_cast<uint32_t>(params.size()) }, UINT);
    // auto* blk = param_tensor.get_host_data();
    // memcpy(blk->ptr, params.data(), sizeof(uint32_t) * params.size());

    // param_tensor.sync();

    // conv2d(param_tensor, inpt, wght, oupt);
    // oupt.sync(false);1
    // inpt.sync(false);

    // const auto* col = static_cast<float*>(oupt.get_host_data()->ptr);
    // for (size_t i = 0; i < oupt.get_size(); ++i)
    // {
    // 	std::cout << col[i] << " ";
    // }


    tensor mval = arange_t<float>(0, 1024, 1);
    mval.reshape({ 4, 4, 64 });
    tensor rval = tensor::zeros<float>({ 64 });
    reduce_mean(2, mval, rval);

    rval.sync(false);
    float* frval = (float*)rval.get_host_data()->ptr;
    std::cout << std::endl;
    for (uint32_t i = 0; i < rval.get_size(); ++i)
        std::cout << frval[i] << " ";

    

    tensor x = tensor::ones<float>({2, 4, 3, 3});
    tensor y = tensor::zeros<float>({2, 4, 3, 3});
    add(x, x, x);

    uint32_t rows = x.get_shape(0);
    uint32_t cols = x.get_size(2) * x.get_shape(1);
    
    tensor M = tensor::zeros<float>({rows});
    tensor v = tensor::zeros<float>({rows});

    tensor w = tensor::zeros<float>({cols});
    tensor b = tensor::zeros<float>({cols});


    instancenorm(x, M, v, w, b, y);
    
    y.sync(false);
    x.sync(false);
 
    const auto* row = static_cast<float*>(x.get_host_data()->ptr);
    const auto* col = static_cast<float*>(y.get_host_data()->ptr);
    for (size_t i = 0; i < x.get_size(); ++i)
        std::cout << row[i] << " ";
    std::cout << std::endl << std::endl;
    for (size_t i = 0; i < y.get_size(); ++i)
    std::cout << col[i] << " ";


    return 0;
}

