# Core Operators

- [ ] adaptive_avg_pool2d
- [ ] adaptive_avg_pool2d_backward
- [ ] log_softmax
- [ ] batch_norm_legit.no_stats
- [ ] softmax
- [ ] to_copy
- [x] abs
- [x] acos
- [x] add_scalar
- [x] add_tensor
- [ ] addmm
- [x] alias
- [ ] amax
- [ ] amin
- [x] argmax
- [x] argmin
- [ ] as_strided
- [x] asin
- [x] asinh
- [x] atan
- [x] atanh
- [ ] avg_pool2d
- [ ] avg_pool2d_backward
- [x] bit_and
- [x] bit_not
- [x] bit_or
- [x] bit_xor
- [ ] bmm
- [x] cat
- [x] clamp
- [x] clone
- [~] col2im
- [ ] constant_pad
- [ ] convolution
- [ ] convolution_backward
- [x] cos
- [x] cosh
- [x] div_scalar
- [x] div_tensor
- [ ] embedding_dense_backward
- [ ] empty_strided
- [x] eq_scalar
- [x] eq_tensor
- [ ] erf
- [x] exp
- [ ] expand
- [ ] flip
- [x] floor
- [x] fmod
- [x] full
- [ ] gather
- [x] ge_scalar
- [x] ge_tensor
- [x] gelu
- [ ] grid_sampler_2d
- [x] gt_scalar
- [x] gt_tensor
- [x] hardtanh
- [ ] index_select
- [ ] isinf
- [ ] isnan
- [x] le_scalar
- [x] le_tensor
- [x] leaky_relu
- [x] log
- [x] logical_and
- [x] logical_not
- [x] logical_or
- [x] lt_scalar
- [x] lt_tensor
- [ ] max_pool2d_with_indices
- [ ] max_pool2d_with_indices_backward
- [ ] max_pool3d_with_indices
- [x] maximum
- [x] max_dim
- [x] mean_dim
- [x] min_dim
- [x] minimum
- [ ] mm
- [x] mul_scalar
- [x] mul_tensor
- [x] native_batch_norm
- [ ] native_dropout
- [ ] native_group_norm
- [ ] native_group_norm_backward
- [x] native_layer_norm
- [ ] native_layer_norm_backward
- [x] ne_scalar
- [x] ne_tensor
- [x] neg
- [ ] permute
- [x] pow_scalar
- [x] pow_tensor 
- [x] reciprocal
- [ ] reflection_pad2d
- [x] relu
- [x] remainder
- [ ] repeat
- [ ] replication_pad2d
- [ ] replication_pad3d
- [x] rsqrt
- [x] scalar_tensor
- [ ] scatter_add
- [ ] scatter_reduce_two
- [ ] select_int
- [x] sigmoid
- [x] sign
- [x] sin
- [x] sinh
- [ ] slice
- [ ] slice_scatter
- [x] sqrt
- [x] squeeze
- [x] sub_scalar
- [x] sub_tensor
- [ ] dim_intlist
- [x] tanh
- [ ] topk
- [x] unsqueeze
- [ ] upsample_bilinear2d
- [ ] upsample_nearest2d
- [ ] var
- [ ] view
- [ ] where


```     


 ```

# Prims Operators
- [x] abs
- [x] acos
- [x] acosh
- [x] asin
- [x] asinh
- [x] atan
- [x] atanh
- [x] cos
- [x] cosh
- [ ] bessel_i0
- [ ] bessel_i02
- [ ] bessel_i1
- [ ] bessel_i1e
- [ ] bessel_j0
- [ ] bessel_j1
- [x] bitwise_not
- [ ] cbrt
- [x] ceil
- [ ] conj_physical
- [ ] diagamma
- [ ] erf
- [ ] erf_inv
- [ ] erfcx
- [x] exp
- [x] expm1
- [x] exp2
- [x] fill
- [x] floor
- [ ] imag
- [ ] isfinite
- [ ] lgamma
- [x] log
- [x] log1p
- [x] log2
- [x] log10
- [ ] ndtri
- [x] neg
- [ ] real
- [x] reciprocal
- [x] round
- [x] sign
- [x] signbit
- [x] sin
- [x] sinh
- [ ] spherical_bessel_j0
- [x] sqrt
- [x] tan 
- [x] tanh
- [x] trunc
- [x] add
- [x] atan2
- [x] bitwise_and
- [x] bitwise_or
- [x] bitwise_xor
- [x] div
- [x] eq
- [x] fmax
- [x] fmin
- [x] fmod
- [ ] gcd
- [x] ge
- [x] gt
- [ ] hypot
- [ ] igamma
- [ ] igammac
- [x] le
- [x] lt
- [x] maximum
- [x] minimum
- [x] mul
- [x] ne
- [ ] nextafter
- [x] pow
- [x] rsqrt
- [ ] shift_left
- [ ] shift_right_arith
- [x] sub
- [ ] zeta
- [ ] as_strided
- [ ] broadcast_in_dim
- [ ] collapse_view
- [ ] conj
- [ ] slice
- [ ] slice_in_dim
- [ ] split_dim
- [x] squeeze
- [x] transpose
- [x] view_of
- [ ] as_strided_scatter
- [x] cat
- [x] reshape
- [ ] rev
- [ ] where
- [x] clone
- [ ] convert_element_type
- [ ] device_put
- [ ] item
- [x] max_value"scalar"
- [x] min_vallue"scalar'
- [ ] copy_strided
- [x] copy_to
- [x] resize
- [ ] amax
- [ ] amin
- [ ] prod
- [ ] sum
- [ ] var
- [ ] empty_strided
- [x] scalar_tensor
- [ ] iota
- [ ] svd
- [ ] normal
- [ ] uniform
- [ ] fft_r2c
- [ ] fft_c2c
- [ ] fft_c2r
