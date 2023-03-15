// blas.h : Header file for your target.

#pragma once
#include "../runtime/runtime.h"
#include "../tensor/tensor.h"
#define TILE_DIM 32

struct matmul_parameter
{
	uint32_t m;
	uint32_t n;
	uint32_t k;
};

inline uint32_t set_group_size_x(const matmul_parameter& p)
{
    return align_size(p.m, TILE_DIM) / TILE_DIM;
}

inline uint32_t set_group_size_y(const matmul_parameter& p)
{
    return align_size(p.n, TILE_DIM) / TILE_DIM;
}

std::string matmul_kernel_code = R"(
#define TILE_DIM 32
layout(push_constant) uniform pushBlock {
	uint m;
	uint n;
	uint k;
};

shared float ATile[TILE_DIM][TILE_DIM];
shared float BTile[TILE_DIM][TILE_DIM];

layout(local_size_x = TILE_DIM, local_size_y = TILE_DIM, local_size_z = 1) in;

)";

inline std::string matmul_shader(const std::string& kernel_shader_code, const tensor& t1, const tensor& t2, const tensor& t3)
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
    local_shader = "#version 460\n" + local_shader;

    local_shader += R"(
void main() {
    uint bx = gl_WorkGroupID.x; // workgroupsize
    uint by = gl_WorkGroupID.y;
    uint thrX = gl_LocalInvocationID.x;
    uint thrY = gl_LocalInvocationID.y;
    uint col = gl_GlobalInvocationID.x; // bx * gl_workGroupSize.x + thrX;
    uint row = gl_GlobalInvocationID.y; // by * gl_WorkGroupSize.y + thrY;

    float elementC = 0;

    for (int t = 0; t < (n-1)/TILE_DIM +1; ++t)
    {
        //threads to load matrix A to shared memory
        if(row < m && t*TILE_DIM+thrX < n)
            ATile[thrY][thrX] = )" + shader_tensor[0] + R"([row*n + t*TILE_DIM+thrX];
        else
            ATile[thrY][thrX] = 0.0f;

        //threads to load matrix B to shared memory
        if (t*TILE_DIM+thrY < n && col < k)
            BTile[thrY][thrX] = )" + shader_tensor[1] + R"([(t*TILE_DIM+thrY)*k + col];
        else
            BTile[thrY][thrX] = 0.0f;

        barrier();
        //calculate a partial value of thread element in C
        for (int i = 0; i < TILE_DIM; ++i)
            elementC = fma(ATile[thrY][i], BTile[i][thrX], elementC);
        barrier();
    }
    //copy final element value to the C matrix
    if (row < m && col < k)
        )" + shader_tensor[2] + R"([row*k+col] = elementC;

})";

    return local_shader;
}





inline void matmul(const tensor& t1, const tensor& t2, const tensor& t3)
{
    const matmul_parameter p{ t1.get_shape(0), t1.get_shape(1), t2.get_shape(1) };
    const std::string kernel_code = matmul_shader(matmul_kernel_code, t1, t2, t3);
    k_runtime->make_job<matmul_parameter>("matmul", kernel_code, { t1.get_data(), t2.get_data() , t3.get_data()}, p, set_group_size_x(p), set_group_size_y(p));
}


#define NUM_THREADS 128
#define WARPSIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

//#define BIGGPU
#ifndef BIGGPU
#define BN NUM_THREADS
#define BM 64
#define BK 16

#define WN 64
#define WM 32
#define TN 4
#define TM 4

#define WNINTER 1
#else 
#define BN NUM_THREADS
#define BM NUM_THREADS
#define BK 16

#define WN 64
#define WM 64
#define TN 4
#define TM 8

#define WNINTER 4
#endif


#define WMINTER (WM * WN) / (WARPSIZE * TM * TN * WNINTER)

#define WSUBN WN / WNINTER
#define WSUBM WM / WMINTER

#define NUM_WARPS NUM_THREADS / WARPSIZE

struct gemm_parameter {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    float alpha;
    float beta;
};

inline uint32_t set_group_size_x(const gemm_parameter& p)
{
    return CEIL_DIV(p.n, BN);
}

inline uint32_t set_group_size_y(const gemm_parameter& p)
{
    return CEIL_DIV(p.m, BM);
}

std::string gemm_kernel_code = R"(
layout(local_size_x = NUM_THREADS) in;
layout(push_constant) uniform pushBlock {
    uint m;
    uint k;
    uint n;
    float alpha;
    float beta;
};

)";

inline std::string gemm_shader(const std::string& kernel_shader_code, const tensor &t1, const tensor &t2, const tensor &t3){
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
    
    std::string out_string;
    gen_type(t3.get_type(), out_string);

    std::string out_pack_string;
    gen_pack(t3.get_type(), out_pack_string, 4);


    local_shader = "#version 460\n" + local_shader + R"(

)" + out_string + R"( regM[WMINTER * TM];
)" + out_string + R"( regN[WNINTER * TN];
)" + out_string + R"( threadResults[WMINTER * TM * WNINTER * TN];

shared )" + out_string + R"( As[BM * BK];
shared )" + out_string + R"( Bs[BK * BN];



    )" + R"(
void loadRegs(uint rowStrideA, uint rowStrideB, uint a, uint b, uint as, uint bs, uint innerRowA, uint innerColA, uint innerRowB, uint innerColB){
    //loads stuff
    for(uint offset = 0; offset+rowStrideA <= BM; offset += rowStrideA) {
        uint xIdx = a + (innerRowA + offset) * k + innerColA * 4;
        )" + out_pack_string + " tmp = " + out_pack_string + "(" + shader_tensor[0] + "[xIdx + 0], " + shader_tensor[0] + "[xIdx + 1], " + shader_tensor[0] + "[xIdx + 2], " + shader_tensor[0] + R"([xIdx + 3]);
        As[as + ((innerColA * 4) + 0) * BM + innerRowA + offset] = tmp.x;
        As[as + ((innerColA * 4) + 1) * BM + innerRowA + offset] = tmp.y;
        As[as + ((innerColA * 4) + 2) * BM + innerRowA + offset] = tmp.z;
        As[as + ((innerColA * 4) + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for(uint offset = 0; offset+rowStrideB <= BK; offset += rowStrideB) {
        uint mIdx = b + (innerRowB + offset) * n + innerColB * 4;
        )" + out_pack_string + " tmp = " + out_pack_string + "(" + shader_tensor[1] + "[mIdx + 0], " + shader_tensor[1] + "[mIdx + 1], " + shader_tensor[1] + "[mIdx + 2], " + shader_tensor[1] + R"([mIdx + 3]);
        Bs[bs + (innerRowB + offset) * BN + innerColB * 4 + 0] = tmp.x;
        Bs[bs + (innerRowB + offset) * BN + innerColB * 4 + 1] = tmp.y;
        Bs[bs + (innerRowB + offset) * BN + innerColB * 4 + 2] = tmp.z;
        Bs[bs + (innerRowB + offset) * BN + innerColB * 4 + 3] = tmp.w;
    }
}

void processFromSmem(uint as, uint bs, uint warpRow, uint warpCol, uint threadRowWarp, uint threadColWarp){
    for(uint dotIdx = 0; dotIdx < BK; ++dotIdx){
        for(uint wSubRowIdx = 0; wSubRowIdx < WMINTER; ++wSubRowIdx){
            for(uint i = 0; i < TM; ++i)
                regM[wSubRowIdx * TM + i] = As[as + (dotIdx * BM) + (warpRow * WM) + (wSubRowIdx * WSUBM) + (threadRowWarp * TM) + i];
        }

        for(uint wSubColIdx = 0; wSubColIdx < WNINTER; ++wSubColIdx){
            for(uint i = 0; i < TN; ++i)
                regN[wSubColIdx * TN + i] = Bs[bs + (dotIdx * BN) + (warpCol * WN) + (wSubColIdx * WSUBN) + (threadColWarp * TN) + i];
        }

        for(uint wSubRowIdx = 0; wSubRowIdx < WMINTER; ++wSubRowIdx){
            for(uint wSubColIdx = 0; wSubColIdx < WNINTER; ++wSubColIdx){

                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {

                        uint reg_idx = (wSubRowIdx * TM + resIdxM) * (WNINTER * TN) + (wSubColIdx * TN) + resIdxN;
                        threadResults[reg_idx] = fma(regM[wSubRowIdx * TM + resIdxM], regN[wSubColIdx * TN + resIdxN], threadResults[reg_idx]);
                    }
                }
            }
        }
    }
}

void main() {
    uint cRow = gl_WorkGroupID.y;
    uint cCol = gl_WorkGroupID.x;

    uint tid = gl_LocalInvocationID.x;

    uint warpIdx = tid / WARPSIZE;
    uint warpCol = warpIdx % (BN / WN);
    uint warpRow = warpIdx / (BN / WN);

    uint threadIdxWarp = tid % WARPSIZE;
    uint threadColWarp = threadIdxWarp % (WSUBN / TN);
    uint threadRowWarp = threadIdxWarp / (WSUBN / TN);

    uint a = cRow * BM * k;
    uint b = cCol * BN;
    uint c = ((cRow * BM) + (warpRow * WM)) * n + (cCol * BN) + (warpCol * WN);
    

    uint innerRowA = tid / (BK / 4);
    uint innerColA = tid % (BK / 4);
    uint rowStrideA = (NUM_THREADS * 4) / BK;
    
    uint innerRowB = tid / (BN / 4);
    uint innerColB = tid % (BN / 4);
    uint rowStrideB = NUM_THREADS / (BN / 4);


    for(uint blkIdx = 0; blkIdx < k; blkIdx += BK) {
        loadRegs(rowStrideA, rowStrideB, a, b, 0, 0, innerRowA, innerColA, innerRowB, innerColB);
        //memoryBarrierShared();
        barrier();
        processFromSmem(0, 0, warpRow, warpCol, threadRowWarp, threadColWarp);

        a += BK;
        b += BK * n;
       
        barrier();
    }

    for(uint wSubRowIdx = 0; wSubRowIdx < WMINTER; ++wSubRowIdx) {
        for(uint wSubColIdx = 0; wSubColIdx < WNINTER; ++wSubColIdx) {
            uint c_interm = c + (wSubRowIdx * WSUBM) * n + wSubColIdx * WSUBN;

            for(uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for(uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {

                    uint c_idx = c_interm + (threadRowWarp * TM + resIdxM) * n + threadColWarp * TN + resIdxN;
                    )" + out_pack_string + " tmp = " + out_pack_string + "(" + shader_tensor[2] + "[c_idx + 0], " + shader_tensor[2] + "[c_idx + 1], " + shader_tensor[2] + "[c_idx + 2], " + shader_tensor[2] + R"([c_idx + 3]);
                    uint i = (wSubRowIdx * TM + resIdxM) * (WNINTER * TN) + wSubColIdx * TN + resIdxN;
                    
                    tmp.x = fma(alpha, threadResults[i + 0], beta * tmp.x);
                    tmp.y = fma(alpha, threadResults[i + 1], beta * tmp.y);
                    tmp.z = fma(alpha, threadResults[i + 2], beta * tmp.z);
                    tmp.w = fma(alpha, threadResults[i + 3], beta * tmp.w);
                    
                    )" + shader_tensor[2] + R"([c_idx + 0] = tmp.x;
                    )" + shader_tensor[2] + R"([c_idx + 1] = tmp.y;
                    )" + shader_tensor[2] + R"([c_idx + 2] = tmp.z;
                    )" + shader_tensor[2] + R"([c_idx + 3] = tmp.w;                    
                }
            }
        }
    }
   
})";

    return local_shader;
}

inline void gemm(const tensor& t1, const tensor& t2, const tensor& t3, const float a, const float b)
{  
    const gemm_parameter p{ t1.get_shape(0), t1.get_shape(1), t2.get_shape(1), a, b};
    std::string kernel_code = gemm_kernel_code;
    kernel_code =  "#define NUM_THREADS " + std::to_string(NUM_THREADS) + "\n" +
                    "#define WARPSIZE " + std::to_string(WARPSIZE) + "\n" + 
                    "#define BN NUM_THREADS\n" + 
                    "#define BM " + std::to_string(BM) + "\n" + 
                    "#define BK " + std::to_string(BK) + "\n" + 
                    "#define WN " + std::to_string(WN) + "\n" + 
                    "#define WM " + std::to_string(WM) + "\n" + 
                    "#define TN " + std::to_string(TN) + "\n" + 
                    "#define TM " + std::to_string(TM) + "\n" +
                    "#define WNINTER " + std::to_string(WNINTER) + "\n" + 
                    "#define WMINTER " + std::to_string(WMINTER) + "\n" + 
                    "#define WSUBN " + std::to_string(WSUBN) + "\n" + 
                    "#define WSUBM " + std::to_string(WSUBM) + "\n" +                     
                    "#define NUM_WARPS " + std::to_string(NUM_WARPS) + "\n" +
                    kernel_code;

    kernel_code = gemm_shader(kernel_code, t1, t2, t3);    
    //std::cout << kernel_code << std::endl;
    k_runtime->make_job<gemm_parameter>("gemm", kernel_code, { t1.get_data(), t2.get_data() , t3.get_data()}, p, set_group_size_x(p), set_group_size_y(p));
}
