import random
import numpy as np


# Constants
# Reconstruction할 Image의 크기
# Ex) 128*64 image면 2^7*2^6이므로 exp_x=7, exp_y=6
exp_x = 2
exp_y = 3
img_size_x = pow(2, exp_x)
img_size_y = pow(2, exp_y)


# num을 이진수로 표현했을 때 1의 개수 세기
def count_number_of_1(num):
    cnt = 0
    while num != 0:
        num &= (num - 1)
        cnt = cnt + 1
    return cnt


# 하다마드 행렬 H_n 생성
def hadamard_matrix(n):
    # Use a breakpoint in the code line below to debug your script.
    size = pow(2, n)
    hadamard = np.ones([size, size])
    for i in range(0, size):
        for j in range(0, size):
            if count_number_of_1(i & j) % 2 == 1:
                hadamard[i][j] = -1
    return hadamard * pow(2, -n/2)


# 2D 하다마드 변환
def hadamard_2d_transform(mat):
    result = np.zeros([img_size_x, img_size_y])
    for i in range(0, img_size_x):
        for j in range(0, img_size_y):
            result[:, j] = mat[:, j] @ hadamard_matrix(exp_x)

    for i in range(0, img_size_x):
        for j in range(0, img_size_y):
            result[i, :] = mat[i, :] @ hadamard_matrix(exp_y)

    return result


# 2D 하다마드 역변환
def hadamard_2d_transform_inv(mat):
    result = np.zeros([img_size_x, img_size_y])
    for i in range(0, img_size_x):
        for j in range(0, img_size_y):
            result[:, j] = mat[:, j] @ np.linalg.inv(hadamard_matrix(exp_x))

    for i in range(0, img_size_x):
        for j in range(0, img_size_y):
            result[i, :] = mat[i, :] @ np.linalg.inv(hadamard_matrix(exp_y))

    return result


img = np.ones([img_size_x, img_size_y])
for i in range(0, img_size_x):
    for j in range(0, img_size_y):
        img[i][j] = random.randint(1, 5)

print("원본 이미지")
print(img)

delta_matrix = np.zeros([img_size_x, img_size_y])
reconstructed_img_hadamard = np.zeros([img_size_x, img_size_y])
for i in range(0, img_size_x):
    for j in range(0, img_size_y):
        delta_matrix[i][j] = 1
        print("델타 행렬")
        print(delta_matrix)

        delta_hadamard_inv = hadamard_2d_transform_inv(delta_matrix)
        print("하다마드 역변환 적용한 델타 행렬")
        print(delta_hadamard_inv)

        mask1 = 0.5 * (delta_hadamard_inv + 1)
        mask2 = 0.5 * (delta_hadamard_inv * (-1) + 1)
        print("마스크1")
        print(mask1)
        print("마스크2")
        print(mask2)

        img_reflected_1 = img * mask1
        img_reflected_2 = img * mask2
        print("mask1을 통과해 나온 이미지")
        print(img_reflected_1)
        print("mask2를 통과해 나온 이미지")
        print(img_reflected_2)

        intensity_1 = np.sum(img_reflected_1)
        intensity_2 = np.sum(img_reflected_2)
        print("mask1을 통과해 나온 이미지의 intensity")
        print(intensity_1)
        print("mask2를 통과해 나온 이미지의 intensity")
        print(intensity_2)

        delta_matrix[i][j] = 0
        reconstructed_img_hadamard[i][j] = intensity_1 - intensity_2
        print("최종 intensity")
        print(intensity_1 - intensity_2)

print("Hadamard transformed reconstructed image")
print(np.round_(reconstructed_img_hadamard, 2))
print("Hadamard transformed original image")
print(np.round_(hadamard_2d_transform(img), 2))

print("Reconstructed image")
print(np.round_(hadamard_2d_transform_inv(reconstructed_img_hadamard), 2))
print("Original image")
print(img)
