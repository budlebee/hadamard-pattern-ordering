# %%
import random
import numpy as np
from math import log2, floor
from scipy.linalg import hadamard
from skimage.transform import resize
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import os


class HadamardPattern():
    def create_normal_order_hmasks(self, imgsize):
        masks = []
        H = hadamard(imgsize**2)
        #H = np.array((H+1)/2)
        for i in tqdm(range(imgsize**2), desc='generating normal order hadamard mask patterns...'):
            mask = np.zeros([imgsize, imgsize])
            for j in range(imgsize**2):
                row = floor(j/imgsize)
                col = j % imgsize
                mask[row][col] = H[i][j]
            mask = np.transpose(np.array(mask))
            masks.append(mask)
        return masks

    def create_cc_order_hmasks(self, imgsize):
        normals = self.create_normal_order_hmasks(imgsize)
        cc = []
        for ii in tqdm(range(len(normals)), desc='generating cake-cutting order hadamard mask patterns...'):
            cc.append((self.get_num_of_chunk(normals[ii]), normals[ii]))
        cc.sort(key=lambda item: item[0])
        return [item[1] for item in cc]

    def check_chunk(self, mat, x, y, visited):
        # out of index check
        if visited.sum() == len(visited)*len(visited[0]):
            return
        if x > len(visited)-1 or y > len(visited[0])-1:
            return
        # double visit check
        if visited[x][y] == 1:
            return
        visited[x][y] = 1
        # x dir moving
        if x != len(mat)-1:
            if mat[x][y] == mat[x+1][y] and visited[x+1][y] == 0:
                self.check_chunk(mat, x+1, y, visited)

        # y dir moving
        if y != len(mat[0])-1:
            if mat[x][y] == mat[x][y+1] and visited[x][y+1] == 0:
                self.check_chunk(mat, x, y+1, visited)

    def get_num_of_chunk(self, mat):
        num_row = len(mat)
        num_col = len(mat[0])
        visited = np.zeros([num_row, num_col])
        count = 0
        idx = 0
        while idx != len(visited)*len(visited[0]):
            if visited[floor(idx/num_row)][idx % num_col] == 0:
                self.check_chunk(mat, floor(idx/num_row),
                                 idx % num_col, visited)
                count += 1

            idx += 1
        return count


# %% normal order reconstruction
imgsize = 32

# load image, convert to grayscale, store as array (matrix)
ground_obj = np.asarray(Image.open("./img/lenna.png").convert('L'))
# Choose new size (don't go higher than 128x128 or Hadamard will kill you)

# Resize image to smaller size for simulation
test_obj = resize(ground_obj, (imgsize, imgsize))
hp = HadamardPattern()
masks = hp.create_normal_order_hmasks(imgsize)

rtn = np.zeros([imgsize, imgsize])
#sample_num = floor(len(masks)/2)
sample_num = (len(masks))
for i in tqdm(range(sample_num), desc="reconstructing image..."):
    projection = np.multiply(test_obj, masks[i])
    rtn = rtn + masks[i] * projection.sum()

# 특정 성분에 값이 집중되는게 문제네.
# 그렇다고 해당 픽셀만 하드코딩해서 처리하기엔, 샘플레이트에 따라서 편중되는 픽셀이 달라진다.
# 플로팅을 할때, 중간값 근처만 그리면 될듯한데.


mean = rtn.mean()
std = rtn.std()
for i in range(imgsize):
    for j in range(imgsize):
        if (rtn[i][j] > mean + std):
            rtn[i][j] = mean
plt.figure()
plt.imshow(rtn)


# %% cc_order reconstruction

# load image, convert to grayscale, store as array (matrix)
# Choose new size (don't go higher than 128x128 or Hadamard will kill you)

# Resize image to smaller size for simulation
masksize = 64
hp = HadamardPattern()
masks = hp.create_cc_order_hmasks(masksize)
# %%


def zoom_matrix(mat, zoom_ratio):
    # matrix 를 zoom_ratio 만큼 확대. zoom ratio 는 정수.
    # zoom ratio 가 2면, 1칸이 2x2 칸으로 확대.
    row = len(mat)
    col = len(mat[0])
    zoomed = np.zeros([row*zoom_ratio, col*zoom_ratio])

    for x in range(row):
        for y in range(col):
            for p in range(zoom_ratio):
                for q in range(zoom_ratio):
                    zoomed[x+p][y+q] = mat[x][y]
    return zoomed


# 10% sampling
imgsize = 64*2
img_set = []
obj_set = []
img_set.append(np.asarray(Image.open("./img/cameraman.png").convert('L')))
img_set.append(np.asarray(Image.open("./img/ghost.png").convert('L')))
img_set.append(np.asarray(Image.open("./img/lenna.png").convert('L')))
for item in img_set:
    obj = resize(item, (imgsize, imgsize))
    obj_set.append(obj)
    rtn = np.zeros([imgsize, imgsize])
    ori_num = (len(masks))
    for i in tqdm(range(ori_num), desc="reconstructing original image..."):
        mask = zoom_matrix(masks[i], 2)
        projection = np.multiply(obj, mask)
        rtn = rtn + mask * projection.sum()
    mean = rtn.mean()
    std = rtn.std()
    plt.figure()
    plt.imshow(rtn)
    rtn = np.zeros([imgsize, imgsize])
    sample_num = floor(len(masks)/10)
    for i in tqdm(range(sample_num), desc="reconstructing 10{%} sampled image..."):
        mask = zoom_matrix(masks[i], 2)
        projection = np.multiply(obj, mask)
        rtn = rtn + mask * projection.sum()
    plt.figure()
    plt.imshow(rtn)
plt.figure()
print('done')

# 33% sampling
imgsize = 64*4
img_set = []
obj_set = []
img_set.append(np.asarray(Image.open("./img/cameraman.png").convert('L')))
img_set.append(np.asarray(Image.open("./img/ghost.png").convert('L')))
img_set.append(np.asarray(Image.open("./img/lenna.png").convert('L')))
for item in img_set:
    obj = resize(item, (imgsize, imgsize))
    obj_set.append(obj)
    rtn = np.zeros([imgsize, imgsize])
    ori_num = (len(masks))
    for i in range(ori_num):
        projection = np.multiply(obj, masks[i])
        rtn = rtn + masks[i] * projection.sum()
    mean = rtn.mean()
    std = rtn.std()
    plt.figure()
    plt.imshow(rtn)
    rtn = np.zeros([imgsize, imgsize])
    sample_num = floor(len(masks)/3)
    for i in range(sample_num):
        projection = np.multiply(obj, masks[i])
        rtn = rtn + masks[i] * projection.sum()
    plt.figure()
    plt.imshow(rtn)
plt.figure()
print('done')
# %%
ground_obj = np.asarray(Image.open("./img/cameraman.png").convert('L'))
test_obj = resize(ground_obj, (imgsize, imgsize))
rtn = np.zeros([imgsize, imgsize])
ori_num = (len(masks))
for i in range(ori_num):
    projection = np.multiply(test_obj, masks[i])
    rtn = rtn + masks[i] * projection.sum()

mean = rtn.mean()
std = rtn.std()
# for i in range(imgsize):
#    for j in range(imgsize):
#        if (rtn[i][j] > mean + std):
#            rtn[i][j] = mean
plt.figure()
plt.imshow(rtn)

rtn = np.zeros([imgsize, imgsize])
sample_num = floor(len(masks))
#sample_num = (len(masks))
for i in range(sample_num):
    projection = np.multiply(test_obj, masks[i])
    rtn = rtn + masks[i] * projection.sum()

mean = rtn.mean()
std = rtn.std()
# for i in range(imgsize):
#    for j in range(imgsize):
#        if (rtn[i][j] > mean + std):
#            rtn[i][j] = mean
plt.figure()
plt.imshow(rtn)
# plt.figure()
# plt.imshow(rtn)


# %% save masks data in json


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


dumped = json.dumps(masks, cls=NumpyEncoder)
path = "/Users/zowan/Documents/python/hadamard-spi/cc_masks128.json"
#path = os.path.dirname(os.path.abspath(__file__))

with open(path, 'w') as f:
    json.dump(dumped, f)

# %%
