import math
import numpy as np


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



x = np.arange(0, 18).astype(np.float32)
# x2 = np.arange(0, 18).astype(np.float32)
o = darknet_img2col(x, channels=2, height=3, width=3, ksize=2, stride=1, pad=0)
gt = o.ravel()




def ksize(dim, params):
    return params[dim]
def stride(dim, ndims, params):
    return params[ndims + dim]
def padding(dim, ndims, params):
    return params[ndims*2 + dim]
def dilation(dim, ndims, params):
    return params[ndims*3 + dim]
def inp(dim, ndims, params, ch=2):
    return params[ndims*4 + dim + ch]

def batch(ndims, params):
    return params[ndims*4 + 0]
def channel(ndims, params):
    return params[ndims*4 + 1]

def out(dim, ndims, params):
    return params[ndims*5 + dim]
def output(dim, ndims, params):
    return int((inp(dim, ndims, params) + 2 * padding(dim, ndims, params) - dilation(dim, ndims, params) * (ksize(dim, params)-1) - 1) / stride(dim, ndims, params)) + 1
    


def OffsetTransform(params, ndims, offset_b, offset_a):

    j = 1
 
    i1 = offset_b % out(2*ndims-1, ndims, params)
    i2 = offset_a % out(ndims-1, ndims, params)

    o1 = math.floor(offset_b / out(2*ndims-1, ndims, params))
    o2 = math.floor(offset_a / out(ndims-1, ndims, params))
    o3 = math.floor(offset_a / ksize(ndims-1, params))
    
    o = i1 * stride(ndims-1, ndims, params) + i2 * dilation(ndims-1, ndims, params) - padding(ndims-1, ndims, params)
    

    for i in range(ndims-1, 0, -1):
        j *= inp(i, ndims, params)
        i1 = o1 % out(i + ndims - 1, ndims, params)
        i2 = o2 % out(i-1, ndims, params)

        o1 = math.floor(o1 / out(i-1 + ndims, ndims, params)) 
        o2 = math.floor(o2 / out(i-1, ndims, params))
        o3 = math.floor(o3 / ksize(i-1, params))

        os =  i1 * stride(i-1, ndims, params) + i2 * dilation(i-1, ndims, params) - padding(i-1, ndims, params)
        o += os * inp(i-1, ndims, params)

    o = o +  o3 * j
    return o

    # off1 = offset_b
    # off2 = offset_a # PROBLEM CHILD
    # off3 = offset_a 
    # ot = 0 

    # start = True
    # for i in range(ndims - 1, -1, -1):   
    #     j *= inp(i, ndims, params)
        
    #     index_a = off1 % out(ndims+i, ndims, params)
    #     index_b = off2 % out(i, ndims, params)

    #     off1 = off1 // out(ndims+i, ndims, params)
    #     off2 = off2 // out(i, ndims, params)
    #     off3 = off3 // ksize(i, params)
    #     offset = index_a * stride(i, ndims, params) + index_b * dilation(i, ndims, params) - padding(i, ndims, params)
        

    #     if offset < 0:
    #         return -1
            
    #     if start:
    #         ot += offset
    #         start = False
    #     else:
    #         ot += offset * inp(i, ndims, params)    

    # ot = ot + off3 * j

    # return ot





def ncol(params, ndims, x, y):
    dim1 = y.shape[-2] * y.shape[-3]
    dim2 = y.shape[-1] 
    X = x.ravel()
    Y = y.ravel()
    for i in range(dim1): # kc
        for j in range(dim2): # ob
            out_offset = i * dim2 + j
            in_offset = OffsetTransform(params, ndims, i, j)
            if gt[out_offset] != X[in_offset]:
                print(i, j, out_offset, in_offset)
            if in_offset == -1:
                Y[out_offset] = 0
            else:
                Y[out_offset] = X[in_offset]
        
    return y

# def coln(params, ndims, x, y):
#     dim1 = x.shape[-2] * x.shape[-3]
#     dim2 = x.shape[-1]

#     X = x.ravel()
#     Y = y.ravel()
#     for i in range(dim1):
#         for j in range(dim2):
#             in_offset = i * dim2 + j
#             out_offset = OffsetTransform(params, ndims, i, j)
#             Y[out_offset] += X[in_offset]
            
#     y = Y.reshape(y.shape)
#     return y


def params(ndims, ksize, stride, padding, dilation):
    def ret(v):
        if isinstance(v, int):
            return ndims*[v]
        else:
            return v
    return ret(ksize) + ret(stride) + ret(padding) + ret(dilation)

default_params = dict(ksize=2, stride=1, padding=0, dilation=1)

def test2d(x):
    if len(x.shape) == 4:
        _injest_dim = 2
    elif len(x.shape) == 3:
        _injest_dim = 1
    else:
        _injest_dim = 0
    
    channel = x.shape[1]
    in_shape = list(x.shape)[_injest_dim:]
    ndims = len(x.shape)-_injest_dim
    
    _params = params(ndims, **default_params) + list(x.shape)
    out_shape = [ksize(dim, _params) for dim in range(ndims)] + [output(dim, ndims, _params) for dim in range(ndims)]
    _params += out_shape

    out_dims =  [channel] + [math.prod(out_shape[:ndims]), math.prod(out_shape[ndims:])]
    
    y1 = np.zeros(out_dims) 
    y2 = ncol(_params, len(in_shape), x, y1)  

    x1 = np.zeros_like(x)  
    # y = coln(_params, len(in_shape), y2, x1)

    return y2

x = np.arange(0, 18).reshape((1,2,3,3)).astype(np.float32)
y = test2d(x)
print(y, '\n')


_p = y.ravel()
for i in range(_p.size):
    print(f' index: {i}; truth: {gt[i]}; value: {_p[i]} ::: delta {gt[i] - _p[i]} ')
    

