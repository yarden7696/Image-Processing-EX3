import sys
from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """


    # kernel_x = np.array([[-1, 0, 1]])
    # kernel_y = kernel_x.T
    # Ix = cv2.filter2D(im2, -1, kernel_x)  # The derivative of I2 in the x-axis
    # Iy = cv2.filter2D(im2, -1, kernel_y)  # # The derivative of I2 in the y-axis
    # It = im2 - im1  # The derivative of I by t
    #
    # orgnlPoints = []
    # u_v = []
    #
    # for i in range(step_size, im1.shape[0], step_size):
    #     for j in range(step_size, im1.shape[1], step_size):
    #         try:
    #             windowIx = Ix[i - win_size // 2:i + 1 + win_size // 2, j - win_size // 2: j + 1 + win_size // 2]
    #             windowIy = Iy[i - win_size // 2:i + 1 + win_size // 2, j - win_size // 2: j + 1 + win_size // 2]
    #             windowIt = It[i - win_size // 2:i + 1 + win_size // 2, j - win_size // 2: j + 1 + win_size // 2]
    #             if windowIx.size < win_size * win_size:
    #                 break
    #             A = np.concatenate(
    #                 (windowIx.reshape((win_size * win_size, 1)), windowIy.reshape((win_size * win_size, 1))), axis=1)
    #             b = (windowIt.reshape((win_size * win_size, 1)))
    #             g, _ = LA.eig(np.dot(A.T, A))
    #             g = np.sort(g)
    #             print(g)
    #             if (g[1] >= g[0] > 1 and (g[1] / g[0]) < 100):
    #                 v = np.dot(np.dot(inv(np.dot(A.T, A)), A.T), b)
    #                 orgnlPoints.append(np.array([j, i]))
    #                 u_v.append(v)
    #                 print(g[0], g[1])
    #
    #         except IndexError as e:
    #             pass
    # return np.array(orgnlPoints), np.array(u_v)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pass


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pyramidRes = []
    h = (2**levels) * (img.shape[0] // (2**levels))
    w = (2**levels) * (img.shape[1] // (2**levels))

    img = cv2.resize(img, (w, h))  # resize the image
    pyramidRes.append(img)

    copyImg = img.copy()
    for i in range(1, levels):
        # Blurring the image with a Gaussian kernel
        sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
        guassian = cv2.getGaussianKernel(5, sigma)
        guassian = guassian * guassian.transpose()
        copyImg = cv2.filter2D(copyImg, -1, guassian, borderType=cv2.BORDER_REPLICATE)

        copyImg = copyImg[::2, ::2]  # Samples every second pixel
        pyramidRes.append(copyImg)
        copyImg = copyImg.copy()
    return pyramidRes

    pass


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    # First we need to check if it is RGB or gray scale and then we will expand the image 2 times by adding
    # zeros (for every two pixels we add 0 after it)
    if img.ndim == 2:  # gray scale case
        zero_padding = np.zeros((2 * img.shape[0], 2 * img.shape[1]), dtype=img.dtype)
    else:  # RGB case
        zero_padding = np.zeros((2 * img.shape[0], 2 * img.shape[1], img.shape[2]), dtype=img.dtype)

    zero_padding[::2, ::2] = img
    return cv2.filter2D(zero_padding, -1, gs_k, borderType=cv2.BORDER_REPLICATE)
    pass


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    pass
