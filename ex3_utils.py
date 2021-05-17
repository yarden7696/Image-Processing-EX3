import sys
from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

"""--------------------------------------------Ex3-----------------------------------------------------"""


def fix(img, levels) -> np.ndarray:
    h = pow(2, levels) * (img.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img.shape[1] // pow(2, levels))
    return img[:h, :w]


def get_Gauss_Kernel(krnlSize: int):
    sigma = 0.3 * ((krnlSize - 1) * 0.5 - 1) + 0.8
    gaussKernel = cv2.getGaussianKernel(krnlSize, sigma)
    gaussKernel = gaussKernel * gaussKernel.transpose()
    gaussKernel *= 4
    return gaussKernel


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    u_v = []
    orgPoints = []

    Gx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    Ix = cv2.filter2D(im2, -1, Gx, borderType=cv2.BORDER_REPLICATE)  # The derivative of I2 in the x-axis
    Iy = cv2.filter2D(im2, -1, Gx.transpose(), borderType=cv2.BORDER_REPLICATE)  # The derivative of I2 in the y-axis
    It = im2 - im1  # The derivative of I by t

    l = 0
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):

            x = Ix[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2:j + win_size // 2 + 1].flatten()
            y = Iy[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2:j + win_size // 2 + 1].flatten()
            t = It[i - win_size // 2:i + win_size // 2 + 1, j - win_size // 2:j + win_size // 2 + 1].flatten()

            b = np.array([[-1 * sum(x[k] * t[k] for k in range(len(x))),
                           -1 * sum(y[k] * t[k] for k in range(len(y)))]]).reshape(2, 1)

            A = np.array([[sum(x[k] ** 2 for k in range(len(x))), sum(x[k] * y[k] for k in range(len(x)))],
                          [sum(x[k] * y[k] for k in range(len(x))), sum(y[k] ** 2 for k in range(len(y)))]])

            ev1, ev2 = np.linalg.eigvals(A)
            if ev2 < ev1:  # sort
                temp = ev1
                ev1 = ev2
                ev2 = temp
            if ev2 / ev1 < 100 and ev2 >= ev1 > 1:
                vel = np.dot(np.linalg.pinv(A), b)  # minimize ||Ad-b||^2
                u = vel[0][0]
                v = vel[1][0]
                u_v.append(np.array([u, v]))
            else:
                l += 1
                u_v.append(np.array([0.0, 0.0]))
            orgPoints.append(np.array([j, i]))
    return np.array(orgPoints), np.array(u_v)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    laplcn_pyrmd = []
    gauss_pyrmd = gaussianPyr(img, levels)  # creating gaussian pyramid with the num of levels we got
    gauss_ker = get_Gauss_Kernel(5)

    for i in range(levels - 1):
        afterExpand = gaussExpand(gauss_pyrmd[i + 1], gauss_ker)  # expand the image in i+1 position from gauss_pyrmd
        imgLplcan = gauss_pyrmd[i] - afterExpand  # Create the difference image (subtraction between Gx-Expand (Gx + 1))
        laplcn_pyrmd.append(imgLplcan.copy())
    laplcn_pyrmd.append(gauss_pyrmd[-1])
    return laplcn_pyrmd

    pass


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """

    # reverse to the Laplacian pyramid so that the first element will is the smallest image
    lap_pyr.reverse()
    lap_img = lap_pyr[0]  # the smallest laplacian image
    gaussKernel = get_Gauss_Kernel(5)

    for i in range(1, len(lap_pyr)):
        afterExpand = gaussExpand(lap_img, gaussKernel)  # expand the lap_img
        lap_img = afterExpand + lap_pyr[i]  # add between the difference image and the expand image

    lap_pyr.reverse()
    return lap_img

    pass


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
# To create a Gaussian pyramid in the first step i blur the image with a Gaussian kernel and second i sampled every
    # second pixel
    pyramidRes = []
    h = (2**levels) * (img.shape[0] // (2**levels))
    w = (2**levels) * (img.shape[1] // (2**levels))
    dim = (w, h)

    img = cv2.resize(img[:h, :w], dim)  # resize the image
    pyramidRes.append(img)  # adding the first image
    copyImg = img.copy()

    for i in range(1, levels):
        # Blurring the image with a Gaussian kernel
        sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
        gaussian = cv2.getGaussianKernel(5, sigma)
        gaussian = gaussian * gaussian.transpose()
        copyImg = cv2.filter2D(copyImg, -1, gaussian, borderType=cv2.BORDER_REPLICATE)

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

    zero_padding[::2, ::2] = img  # Samples every second pixel
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

    gaussKernel = get_Gauss_Kernel(5)
    mask = fix(mask, levels)
    img_1 = fix(img_1, levels)
    img_2 = fix(img_2, levels)

    n_blend = img_1 * mask + (1 - mask) * img_2  # naive blending

    Gm = gaussianPyr(mask, levels)  # Build a Gaussian pyramid for mask
    La = laplaceianReduce(img_1, levels)  # Build a Laplacian pyramid for img_1
    Lb = laplaceianReduce(img_2, levels)  # Build a Laplacian pyramid for img_2

    Lc = La[levels - 1] * Gm[levels - 1] + (1 - Gm[levels - 1]) * Lb[levels - 1]
    for i in range(levels - 2, -1, -1):
        # Adding between a laplacian image at level i and a laplacian image with expand at level i + 1
        Lc = gaussExpand(Lc,gaussKernel) + La[i] * Gm[i] + (1 - Gm[i]) * Lb[i]

    return n_blend, Lc

    pass
