
import math
import numpy as np


def im2col_get_pixel(im, height, width, channel, row, col):
  
    if row < 0 or col < 0 or row >= height or col >= width:
        return 0
    return im[int(col + width * (row + height * channel))] # batch*w*h*c + width*height*channel + width*row + col


def darknet_img2col(data, channels, height, width, ksize, stride, pad):
    out_h = int((height + 2*pad - ksize) / stride) + 1
    out_w = int((width + 2*pad - ksize) / stride) + 1

    channels_cols = channels*ksize*ksize
    out_shape = (channels_cols, out_h*out_w)
    elem_cnt = out_shape[0] * out_shape[1]
    out_array = np.zeros(shape=elem_cnt, dtype=np.float32)

    for c in range(channels_cols):
        # Calculate the offset of h, w in a window of k*k respectively
        kh_offset = (c // ksize) % ksize
        kw_offset = c % ksize
        # Calculate the channel index of the current process
        c_im = c // ksize // ksize
        for h in range(out_h):
            for w in range(out_w):
                im_row = h * stride + kh_offset - pad
                im_col = w * stride + kw_offset - pad
                index = (c * out_h + h) * out_w + w
                out_array[index] = im2col_get_pixel(data, height, width, c_im, im_row, im_col)

    out_array = np.reshape(out_array, out_shape)

    return out_array



# x = np.arange(0, 9).astype(np.float32)
# out = darknet_img2col(x, channels=1, height=3, width=3, ksize=2, stride=1, pad=0)
# print(out)
# print()



def ksize(dim, params):
    return params[dim]
def stride(dim, ndims, params):
    return params[ndims + dim]
def padding(dim, ndims, params):
    return params[ndims*2 + dim]
def dilation(dim, ndims, params):
    return params[ndims*3 + dim]
def input(dim, ndims, params):
    _cb = params[-1]
    return params[ndims*4 + dim]
def out(dim, ndims, params):
    _cb = params[-1]
    return params[ndims*5 + dim]

def channels(ndims, params):
    return params[ndims*4 + 1]

def output(input_shape, dim, ndims, params):
    return int((input_shape[dim] + 2 * padding(dim, ndims, params) - dilation(dim, ndims, params) * (ksize(dim, params)-1) - 1) / stride(dim, ndims, params)) + 1
    


def OffsetTransform(params, ndims, offset_a, offset_b):
    off1 = offset_b
    off2 = offset_a
    off3 = offset_a
    start = True
    j = 1
    ot = 0 
    
    for i in range(ndims - 1, -1, -1):
        index_a = off1 % out(ndims+i+2, ndims, params)
        index_b = off2 % out(i+2, ndims, params)
        off1 = math.floor(off1 / out(ndims+i+2, ndims, params))
        off2 = math.floor(off2 / out(i+2, ndims, params))
        off3 = math.floor(off3 / ksize(i, params))

        j *= input(i+2, ndims, params)

        offset = index_a * stride(i, ndims, params) +  index_b * dilation(i, ndims, params) - padding(i, ndims, params)
        
        if start:
            ot += offset
            start = False
        else:
            ot += offset * input(i+2, ndims, params)

    ot = ot + off3 * j
    return ot
 

def OffsetTransform2(params, ndims, offset_a, offset_b):
    idx = ndims - 1 + 2
    idx2 = idx - 2

    index_a = offset_b % out(ndims + idx, ndims, params)
    index_b = offset_a % out(idx, ndims, params)

    off_1 = offset_b // out(ndims + idx, ndims, params)
    off_2 = offset_a // out(idx, ndims, params)
    off_3 = offset_a // ksize(idx2, params)

    j = input(idx, ndims, params)
    ot = index_a * stride(idx2, ndims, params) + index_b * dilation(idx2, ndims, params) - padding(idx2, ndims, params)
    
    for idx in range(idx - 1, 1, -1):
        idx2 = idx - 2

        index_a = off_1 % out(ndims + idx, ndims, params)
        index_b = off_2 % out(idx, ndims, params)

        off_1 = off_1 // out(ndims + idx, ndims, params)
        off_2 = off_2 // out(idx, ndims, params)
        off_3 = off_3 // ksize(idx2, params)

        offset = index_a * stride(idx2, ndims, params) + index_b * dilation(idx2, ndims, params) - padding(idx2, ndims, params)
        
        if offset < 0 or offset > input(idx, ndims, params):
            return -1

        j *= input(idx, ndims, params)
        ot += offset * input(idx, ndims, params)

    ot = ot + off_3 * j
    return ot
 

def ncol(params, ndims, x, y):
    dim1 = y.shape[0]
    dim2 = y.shape[1]
        
    X = x.ravel()
    Y = y.ravel()

    for i in range(dim1):
        for j in range(dim2):
            out_offset = i * dim2 + j
            in_offset = OffsetTransform2(params, ndims, i, j)
       
            if in_offset < len(X) and out_offset < len(Y):
                Y[out_offset] = X[in_offset]

    y = Y.reshape(y.shape)
    return y

def coln(params, ndims, x, y):
    dim1 = x.shape[0]
    dim2 = x.shape[1]

    X = x.ravel()
    Y = y.ravel()

    for j in range(dim2):
        for i in range(dim1):
            in_offset = i * dim2 + j
            out_offset = OffsetTransform2(params, ndims, i, j)

            if in_offset < len(X) and out_offset < len(Y):
                Y[out_offset] += X[in_offset]
        
    y = Y.reshape(y.shape)
    return y    
 
def unfold(default_params, x):
    in_shape = list(x.shape)[2:]
    batch_size = x.shape[0]
    ndims = len(x.shape)-2
    _params = params(ndims, **default_params) + list(x.shape)
    out_shape = [ksize(dim, _params) for dim in range(ndims)] + \
        [output(in_shape, dim, ndims, _params) for dim in range(ndims)]
    _params += out_shape + [2]
    out_dims =  [channels(ndims, _params) * math.prod(out_shape[:ndims]), batch_size * math.prod(out_shape[ndims:])]
    y = np.zeros(out_dims)
    return ncol(_params, len(in_shape), x, y)  

def fold(default_params, x, y):
    in_shape = list(x.shape)[2:]
    batch_size = x.shape[0]
    ndims = len(x.shape)-2
    _params = params(ndims, **default_params) + list(x.shape)
    out_shape = [ksize(dim, _params) for dim in range(ndims)] + \
        [output(in_shape, dim, ndims, _params) for dim in range(ndims)]
    _params += out_shape + [2]
    out_dims =  [channels(ndims, _params) * math.prod(out_shape[:ndims]), batch_size * math.prod(out_shape[ndims:])]
    x1 = np.zeros_like(x)
    return coln(_params, len(in_shape), y, x1)

def test2d(x):    
    in_shape = list(x.shape)[2:]
    batch_size = x.shape[0]
    ndims = len(x.shape)-2

    _params = params(ndims, **default_params) + list(x.shape)
    out_shape = [ksize(dim, _params) for dim in range(ndims)] + \
        [output(in_shape, dim, ndims, _params) for dim in range(ndims)]
    _params += out_shape + [2]
    out_dims =  [channels(ndims, _params) * math.prod(out_shape[:ndims]), batch_size * math.prod(out_shape[ndims:])]
    
    y1 = np.zeros(out_dims)

    x1 = np.zeros_like(x)
   
    y2 = ncol(_params, len(in_shape), x, y1)  
    print(y1.shape) # C KxK B SxS
    print(y1.ravel())
    print()
    y = coln(_params, len(in_shape), y2, x1)
    print(y.ravel())
    return y



def conv(default_params, x, w):
    in_shape = list(x.shape)[2:]
    in_size = x.shape[1] * x.shape[2] * x.shape[3]
    batch_size = x.shape[0]
    ndims = len(x.shape)-2
    _params = params(ndims, **default_params) + list(x.shape)
    out_shape = [ksize(dim, _params) for dim in range(ndims)] + \
        [output(in_shape, dim, ndims, _params) for dim in range(ndims)]
    _params += out_shape + [2]
    out_dims =  [batch_size, channels(ndims, _params) * math.prod(out_shape[:ndims]), math.prod(out_shape[ndims:])]
    u = np.zeros(out_dims)
    y = np.zeros([w.shape[0], batch_size * math.prod(out_shape[ndims:])])
    
    dim1 = u.shape[1]
    dim2 = u.shape[2]
    
    X = x.ravel()
    U = u.ravel()
    W = w.ravel()
    Y = y.ravel()

    print(X.shape, U.shape, W.shape, dim1, dim2)
    for b in range(batch_size): # x N
        for i in range(dim1): # IN_C X KERNEL         
            for j in range(dim2): # OUT 
                out_offset = i * dim2 + j
                in_offset = OffsetTransform2(_params, ndims, i, j)

                for k in range(w.shape[0]):
                    if in_offset < len(X)/batch_size and in_offset > 0:
                        Y[b *dim2 * w.shape[0] + k * dim2 +  j] += W[k * dim1 + i] * X[in_offset + b * in_size]
                        U[b * dim2 * dim1 + out_offset] = X[in_offset + b * in_size] #* W[w_offset]
                    else:
                        Y[b *dim2 * w.shape[0] + k * dim2 +  j] = 0;

    u = U.reshape(u.shape)
    # y = w @ u
    y = Y.reshape(y.shape)
    print("MY CONV SOLUTION", y.shape)

    return y


def params(ndims, ksize, stride, padding, dilation):
    def ret(v):
        if isinstance(v, int):
            return ndims*[v]
        else:
            return v
    return ret(ksize) + ret(stride) + ret(padding) + ret(dilation)


default_params = dict(ksize=2, stride=1, padding=1, dilation=1)


out_channels = 4
x = np.arange(0, 2*7*7).reshape((2,1,7,7)).astype(np.float32)
w = np.ones([out_channels, x.shape[1]] + 2*[default_params['ksize']])
# u = unfold(default_params, x)
w_t = w.reshape([out_channels, -1])
y  = conv(default_params, x, w_t)

# print("my unfold", c.shape)
# print(c.ravel())
# print()
# y = w_t @ u
print("my conv", y.shape)
print(y.ravel())

print(":::")

# fold_params = dict(ksize=1, dilation=1, stride=1, padding=0)
# # f = fold(fold_params, x, y)
# # print(f.ravel())



import torch
import torch.nn as nn


fold_params = dict(kernel_size=default_params["ksize"], dilation=default_params["dilation"], padding=default_params["padding"], stride=default_params["stride"])

pt_x = torch.Tensor(x)
pt_w = torch.Tensor(w)
# unfold = nn.Unfold(**fold_params)
# fold = nn.Fold(output_size = (7, 7), kernel_size=(1,1))


print("PYTORCH")
# pt_u = unfold(pt_x)
# pt_u_t1 = pt_u.transpose(1, 2)
# pt_w2 =  pt_w.view(pt_w.size(0), -1).t()

# # pt_y2_m = pt_u_t1.matmul(pt_w)
# pt_m = pt_u_t1.matmul(pt_w2)

# pt_u_t2 = pt_m.transpose(1, 2)
# pt_y = fold(pt_u_t2)

gt = nn.functional.conv2d(pt_x, pt_w, stride=fold_params['stride'], padding=fold_params['padding'], dilation=fold_params['dilation'])
print("ground truth", gt.shape)
print(gt.detach().numpy().ravel())
# print()
# print("unfold truth", pt_u.shape)
# print(pt_u.detach().numpy().ravel())
# print()
# print("conv truth", pt_y.shape)
# print(pt_y.detach().numpy().ravel())