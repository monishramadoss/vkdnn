template<typename INDEX_T, int NDIM, int SDIM>
bool UnfoldIndexTransform(const UnfoldParams<INDEX_T, NDIM, SDIM>& params,
                                         const INDEX_T* index_a, INDEX_T* index_b) {
  // batch dim index transform
  index_b[0] = index_a[0];
  // channel dim index transform
  using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
  index_b[ParamType::kInputChannelDim] = index_a[ParamType::kOutputChannelDim];
// spatial dim index transform

  // D,H,W spatial dim index transform
  for (int64_t d = 0; d < NDIM; ++d) {
    INDEX_T idx = index_a[SDIM + NDIM + d] * params.stride[d]
                  + index_a[SDIM + d] * params.dilation[d] - params.padding[d];
    if (idx < 0 || idx >= params.dims[d]) return true;
    index_b[SDIM + d] = idx;
  }
  return false;
}


template<typename INDEX_T, int NDIM, int SDIM>
bool FoldIndexTransform(const FoldParams<INDEX_T, NDIM, SDIM>& params,
                                       const INDEX_T* index_a, INDEX_T* index_b) {
  // batch dim index transform
  index_b[0] = index_a[0];
  // channel dim index transform
  using ParamType = FoldParams<INDEX_T, NDIM, SDIM>;
  index_b[ParamType::kInputChannelDim] = index_a[ParamType::kOutputChannelDim];
// spatial dim index transform
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  // D,H,W spatial dim index transform
  for (int64_t d = 0; d < NDIM; ++d) {
    INDEX_T idx = index_a[SDIM + NDIM + d] * params.stride[d]
                  + index_a[SDIM + d] * params.dilation[d] - params.padding[d];
    if (idx < 0 || idx >= params.dims[d]) return true;
    index_b[SDIM + d] = idx;
  }
  return false;
}


template<typename T, typename INDEX_T, int NDIM, int SDIM>
__global__ void CudaUnfoldForward(UnfoldParams<INDEX_T, NDIM, SDIM> params, const T* in, T* out) {
  CUDA_1D_KERNEL_LOOP_T(INDEX_T, out_offset, params.out_elem_cnt) {
    using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
    INDEX_T in_index[ParamType::kInputNDim] = {0};
    INDEX_T out_index[ParamType::kOutputNDim] = {0};
    params.out_index_helper.OffsetToNdIndex(out_offset, out_index);
    if (!UnfoldIndexTransform<INDEX_T, NDIM, SDIM>(params, out_index, in_index)) {
      INDEX_T in_offset = params.in_index_helper.NdIndexToOffset(in_index);
      out[out_offset] = in[in_offset];
    } else {
      out[out_offset] = static_cast<T>(kUnfoldPaddingValue);
    }
  }
}


template<typename T, typename INDEX_T, int NDIM, int SDIM>
__global__ void CudaFoldForward(FoldParams<INDEX_T, NDIM, SDIM> params, const T* input_ptr,
                                T* output_ptr) {
  CUDA_1D_KERNEL_LOOP_T(INDEX_T, in_offset, params.in_elem_cnt) {
    using ParamType = FoldParams<INDEX_T, NDIM, SDIM>;
    INDEX_T in_index[ParamType::kInputNDim] = {0};
    INDEX_T out_index[ParamType::kOutputNDim] = {0};
    params.in_index_helper.OffsetToNdIndex(in_offset, in_index);
    if (!FoldIndexTransform<INDEX_T, NDIM, SDIM>(params, in_index, out_index)) {
      INDEX_T out_offset = params.out_index_helper.NdIndexToOffset(out_index);
      XPUAdd<T>::Invoke(&input_ptr[in_offset], &output_ptr[out_offset]);
    } else {
      continue;
    }
  }



template<typename T>
__inline__ __device__ void WelfordBlockAllReduce(T thread_mean, T thread_m2, T thread_count,
                                                 T* result_mean, T* result_m2, T* result_count) {
  __shared__ T mean_shared[kWarpSize];
  __shared__ T m2_shared[kWarpSize];
  __shared__ T count_shared[kWarpSize];
  __shared__ T mean_result_broadcast;
  __shared__ T m2_result_broadcast;
  __shared__ T count_result_broadcast;
  const int lid = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;
  T warp_mean = 0;
  T warp_m2 = 0;
  T warp_count = 0;
  WelfordWarpReduce(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
  __syncthreads();
  if (lid == 0) {
    mean_shared[wid] = warp_mean;
    m2_shared[wid] = warp_m2;
    count_shared[wid] = warp_count;
  }
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) {
      warp_mean = mean_shared[lid];
      warp_m2 = m2_shared[lid];
      warp_count = count_shared[lid];
    } else {
      warp_mean = static_cast<T>(0);
      warp_m2 = static_cast<T>(0);
      warp_count = static_cast<T>(0);
    }
    __syncwarp();
    T block_mean = 0;
    T block_m2 = 0;
    T block_count = 0;
    WelfordWarpReduce(warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
    if (lid == 0) {
      mean_result_broadcast = block_mean;
      m2_result_broadcast = block_m2;
      count_result_broadcast = block_count;
    }
  }
  __syncthreads();
  *result_mean = mean_result_broadcast;
  *result_m2 = m2_result_broadcast;
  *result_count = count_result_broadcast;
}


template<typename LOAD, typename STORE, typename ComputeType, int pack_size, int block_size>
__global__ void LayerNormBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                           const int64_t cols, const double epsilon,
                                           ComputeType* mean, ComputeType* inv_variance) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        WelfordCombine(pack[i], &thread_mean, &thread_m2, &thread_count);
      }
    }
    
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_count = 0;
    WelfordBlockAllReduce<ComputeType>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2,
                                       &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = Rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      m[row] = row_mean;
      inv_variance[row] = row_inv_var;
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      const int pack_offset = pack_id * pack_size;
      load.template load<pack_size>(pack, row, pack_offset);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) { 
        pack[i] = (pack[i] - row_mean) * row_inv_var; 
      }
      store.template store<pack_size>(pack, row, pack_offset);
    }
  }
}