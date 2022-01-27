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


def make_hadamard_matrix(size):
    # N, N/12, N/20 이 2의 배수인지 체크해야함.
    # print(log2(size)-floor(log2(size)))
    # print(log2(size/12.0)-floor(log2(size/12)))
    # print(log2(size/20.0)-floor(log2(size/20.0)))
    n = floor(log2(size))
    if n-floor(n) != 0:
        print("size is not 2^n")
        return None
    print("size: ", size)
    H2 = np.array([[1, 1], [1, -1]])
    H = np.array([[1, 1], [1, -1]])
    for i in range(n-1):
        H = np.kron(H, H2)
    return np.where(H == -1, 0, H)


def create_normal_order_hmasks(imgsize):
    masks = []
    H = hadamard(imgsize**2)

    H = np.array((H+1)/2)

    for i in tqdm(range(imgsize**2), desc='generating hadamard mask patterns...'):
        mask = np.zeros([imgsize, imgsize])
        for j in range(imgsize**2):
            row = floor(j/imgsize)
            col = j % imgsize
            mask[row][col] = H[i][j]
        mask = np.transpose(np.array(mask))
        masks.append(mask)
    return masks

# %%


imgsize = 128

# load image, convert to grayscale, store as array (matrix)
ground_obj = np.asarray(Image.open("./img/lenna.png").convert('L'))
# Choose new size (don't go higher than 128x128 or Hadamard will kill you)

# Resize image to smaller size for simulation
test_obj = resize(ground_obj, (imgsize, imgsize))

masks = create_normal_order_hmasks(imgsize)

# %% sampling and reconstruction process.
rtn = np.zeros([imgsize, imgsize])
#sample_num = floor(len(masks)/2)
sample_num = (len(masks))
for i in range(sample_num):
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
path = "/Users/zowan/Documents/python/hadamard-spi/masks128.json"
#path = os.path.dirname(os.path.abspath(__file__))

with open(path, 'w') as f:
    json.dump(dumped, f)
# %%
