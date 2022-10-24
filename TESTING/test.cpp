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
