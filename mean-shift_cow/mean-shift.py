import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale
from tqdm import tqdm

torch.set_grad_enabled(False)
torch.set_default_tensor_type(torch.DoubleTensor)

def distance(x, X):
    dist = torch.norm((X - x), p = 2, dim = 1)
    return dist

def distance_batch(x, X):
    x = x.unsqueeze(1)
    X_temp = X.repeat(x.shape[0], 1)
    X_temp = X_temp.reshape(x.shape[0], -1, 3)
    dist = torch.norm((X_temp - x), p=2, dim=2)
    return dist


def gaussian(dist, bandwidth):
    weight = torch.exp(-torch.mul(dist, dist)/ (2 * bandwidth * bandwidth))
    return weight


def update_point(weight, X):
    sum = torch.sum(weight)
    weight = weight.reshape(1, X.shape[0])
    return torch.mm(weight/sum, X)

def update_point_batch(weight, X):
    sum = torch.sum(weight, dim=1)
    sum = sum.reshape(sum.shape[0], 1)
    return torch.mm(weight/sum, X)


def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    batchsize = 2048
    X_ = X.clone()
    data_loader = torch.utils.data.DataLoader(X, batch_size=batchsize, num_workers=0, shuffle=False)
    j = 0
    for x in tqdm(data_loader):
        dist = distance_batch(x, X)
        weight = gaussian(dist, bandwidth)
        X_[j*batchsize:(j+1)*batchsize] = update_point_batch(weight, X)
        j += 1
    return X_


def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X


scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
