import math
import numpy as np

x = np.ones([2, 4, 3, 3, 3])

np_x = x.reshape([2, 4, -1])
eps = 1e-05



def instNorm(x):
    y = np.zeros_like(x)
    m = x.mean(axis=-1)
    v = x.var(axis=-1)

    print("mean", m)
    print("vari", v)

    print("i", m.shape)
    for i in range(x.shape[2]):     
        y[:, :, i] = (x[:, :, i] - m) / np.sqrt(v + eps)
    return y

def layerNorm(x):
    y = np.zeros_like(x)
    m = x.mean(axis=(1, -1))
    v = x.var(axis=(1, -1))
    print("l", m.shape)
    for i in range(x.shape[0]):
        y[i] = (x[i] - m[i]) / np.sqrt(v[i] + eps)
    return y

def batchNorm(x):
    y = np.zeros_like(x)
    m = x.mean(axis=(0, -1))
    v = x.var(axis=(0, -1))
    print("b", m.shape)
    for i in range(x.shape[1]):
        y[:, i, ...] = (x[:, i, ...] - m[i]) / np.sqrt(v[i] + eps)
    return y

print(np_x.shape)
np_l_y = layerNorm(np_x)
np_i_y = instNorm(np_x)
np_b_y = batchNorm(np_x) # working


import torch
import torch.nn as nn
pt_x = torch.Tensor(x)

layer_norm = nn.LayerNorm([4, 3, 3, 3])
batch_norm = nn.BatchNorm3d(4)
insta_norm = nn.InstanceNorm3d(4)

pt_l_y = layer_norm(pt_x)
pt_b_y = batch_norm(pt_x)
pt_i_y = insta_norm(pt_x)

print(pt_l_y.detach().numpy())

import numpy as np


def welford_update(count, mean, M2, currValue):
    count += 1
    delta = currValue - mean
    mean += delta / count
    delta2 = currValue - mean
    M2 += delta * delta2
    return (count, mean, M2)


def naive_update(sum, sum_square, currValue):
    sum = sum + currValue
    sum_square = sum_square + currValue * currValue
    return (sum, sum_square)


x_arr = np.ones(100000).astype(np.float32)

welford_mean = 0
welford_m2 = 0
welford_count = 0
for i in range(len(x_arr)):
    new_val = x_arr[i]
    welford_count, welford_mean, welford_m2 = welford_update(welford_count, welford_mean, welford_m2, new_val)

naive_sum = 0
naive_sum_square = 0
for i in range(len(x_arr)):
    new_val = x_arr[i]
    naive_sum, naive_sum_square = naive_update(naive_sum, naive_sum_square, new_val)
naive_mean = naive_sum / len(x_arr)
naive_var = naive_sum_square/ len(x_arr) - naive_mean*naive_mean

#print(np_i_y.ravel(), '\n')

#print(pt_i_y.detach().numpy().ravel())


