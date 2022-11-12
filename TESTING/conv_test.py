from email.policy import default
import math
import numpy as np
from sklearn.neighbors import DistanceMetric


def im2col_get_pixel(im, height, width, channel, row, col):
  
    if row < 0 or col < 0 or row >= height or col >= width:
        return 0
    return im[int(col + width * (row + height * channel))] # batch*width*height*channel + width*height*channel + width*row + col


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



x = np.arange(0, 9).astype(np.float32)
x2 = np.arange(0, 18).astype(np.float32)
out = darknet_img2col(x2, channels=2, height=3, width=3, ksize=2, stride=1, pad=0)
print(out)
print()



def ksize(dim, params):
    return params[dim]
def stride(dim, ndims, params):
    return params[ndims + dim]
def padding(dim, ndims, params):
    return params[ndims*2 + dim]
def dilation(dim, ndims, params):
    return params[ndims*3 + dim]
def input(dim, ndims, params):
    _cb = params[ndims*4]
    return params[ndims*4 + dim + _cb+1]
def out(dim, ndims, params):
    _cb = params[ndims*4]
    return params[ndims*5 + dim + _cb+1]

def channels(ndims, params):
    _cb = params[ndims*4]
    return params[ndims*4 + _cb]

def output(dim, ndims, params):
    return int((input(dim, ndims, params) + 2 * padding(dim, ndims, params) - dilation(dim, ndims, params) * (ksize(dim, params)-1) - 1) / stride(dim, ndims, params)) + 1
    


def OffsetTransform(params, ndims, offset_a, offset_b, offset_c):
    off1 = offset_b
    off2 = offset_a
    off3 = offset_a
    ot = 0 
    
    start = True
    for i in range(ndims - 1, -1, -1):
        index_a = off1 % out(ndims+i, ndims, params)
        index_b = off2 % out(i, ndims, params)
        off1 = math.floor(off1 / out(ndims+i, ndims, params))
        off2 = math.floor(off2 / out(i, ndims, params))
        off3 = off3 // ksize(i, params)

        offset = index_a * stride(i, ndims, params) + \
            index_b * dilation(i, ndims, params) - padding(i, ndims, params)
        if offset < 0 or offset >= input(i, ndims, params):
            return -1
        if start:
            ot += offset
            start = False
        else:
            ot += offset * input(i, ndims, params)
    return ot + off3 * channels(ndims, params) 
 


def ncol(params, ndims, x, y):
    channel = y.shape[-3]
    dim1 = y.shape[-2]
    dim2 = y.shape[-1]

    X = x.ravel()
    Y = y.ravel()
    for i in range(channel * dim1):
        for j in range(dim2):
            out_offset = i * dim2 +  j
            in_offset = OffsetTransform(params, ndims, i, j, 0)
            if in_offset == -1:
                Y[out_offset] = 0
            else:
                Y[out_offset] = X[in_offset]

    y = Y.reshape(y.shape)
    return y

def coln(params, ndims, x, y):
    channel = x.shape[-3]
    dim1 = x.shape[-2]
    dim2 = x.shape[-1]

    X = x.ravel()
    Y = y.ravel()


    for c in range(channel):
        for i in range(dim1):
            for j in range(dim2):
                in_offset = dim2 * (c * dim1 + i) + j
                out_offset = OffsetTransform(params, ndims, i, j, c)
                Y[out_offset] += X[in_offset]
        
    y = Y.reshape(y.shape)
    return y


def params(ndims, ksize, stride, padding, dilation):
    def ret(v):
        if isinstance(v, int):
            return ndims*[v]
        else:
            return v
    return ret(ksize) + ret(stride) + ret(padding) + ret(dilation)

def index_strides(shape):
    stride = [1 for _ in range(len(shape)+1)]

    for i in range(len(stride)-1, 0, -1):
        stride[i-1] = stride[i] * shape[i-1]
    return stride

default_params = dict(ksize=2, stride=1, padding=0, dilation=1)

def test2d(x):
    if len(x.shape) == 4:
        _injest_dim = 2
    elif len(x.shape) == 3:
        _injest_dim = 1
    
    in_shape = list(x.shape)[_injest_dim:]
    in_strides = index_strides(x.shape)
    ndims = len(x.shape)-_injest_dim
    
    _params = params(ndims, **default_params) + [_injest_dim] + list(x.shape)
    out_shape = [ksize(dim, _params) for dim in range(ndims)] + [output(dim, ndims, _params) for dim in range(ndims)]
    _params += out_shape


    out_dims =  [channels(ndims, _params)] + [math.prod(out_shape[:ndims]), math.prod(out_shape[ndims:])]
    
    y1 = np.zeros(out_dims)
    x1 = np.zeros_like(x)

    y2 = ncol(_params, len(in_shape), x, y1)  
    
    # y = coln(_params, len(in_shape), y2, x1)

    return y2

x1 = np.arange(0, 9).astype(np.float32)
x2 = np.arange(0, 18).reshape((1,2,3,3)).astype(np.float32)

print(test2d(x2))