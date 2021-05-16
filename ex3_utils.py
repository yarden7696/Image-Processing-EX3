import sys
from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

# לשנות עוד את הפונקציה ולכתוב הערות
def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    orgPoints = []
    u_v = []

    Iy = cv2.Sobel(im1, -1, 0, 1)  # The derivative of I2 in the y-axis
    Ix = cv2.Sobel(im1, -1, 1, 0)  # The derivative of I2 in the x-axis
    It = im1 - im2  # The derivative of I by t

    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):
            try:
                winIx = Ix[i - win_size // 2:i + 1 + win_size // 2, j - win_size // 2:j + 1 + win_size // 2]
                winIy = Iy[i - win_size // 2:i + 1 + win_size // 2, j - win_size // 2:j + 1 + win_size // 2]
                winIt = It[i - win_size // 2:i + 1 + win_size // 2, j - win_size // 2:j + 1 + win_size // 2]

                if winIx.size < (win_size * win_size):
                    break
                Amat = np.concatenate(
                    (winIx.reshape(((win_size * win_size), 1)), winIy.reshape(((win_size * win_size), 1))), axis=1)
                bMat = (winIt.reshape(((win_size * win_size), 1)))
                eig, _ = LA.eig(np.dot(Amat.T, Amat))
                eig = np.sort(eig)
                if eig[1] >= eig[0] > 1 and (eig[1] / eig[0]) < 100:
                    d = np.dot(np.dot(inv(np.dot(Amat.T, Amat)), Amat.T), bMat)
                    orgPoints.append(np.array([j, i]))
                    u_v.append(d)

            except IndexError as e:
                pass
    return np.array(orgPoints), np.array(u_v)





def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    laplcn_pyrmd = []

    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    gauss_ker = cv2.getGaussianKernel(5, sigma)
    gauss_ker = gauss_ker * gauss_ker.transpose()
    gauss_ker *= 4

    gauss_pyrmd = gaussianPyr(img, levels)  # creating gaussian pyramid with the num of levels we got
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

    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    gaussKernel = cv2.getGaussianKernel(5, sigma)
    gaussKernel = gaussKernel * gaussKernel.transpose()
    gaussKernel *= 4

    # reverse to the Laplacian pyramid so that the first element will is the smallest image
    lap_pyr.reverse()
    lap_img = lap_pyr[0]  # the smallest laplacian image

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

    zero_padding[::2, ::2] = img  # Samples every second pixel
    return cv2.filter2D(zero_padding, -1, gs_k, borderType=cv2.BORDER_REPLICATE)
    pass





def fix(img, levels) -> np.ndarray:
    h = pow(2, levels) * (img.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img.shape[1] // pow(2, levels))
    return img[:h, :w]

def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """

    sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8
    gaussKernel = cv2.getGaussianKernel(5, sigma)
    gaussKernel = gaussKernel * gaussKernel.transpose()
    gaussKernel *= 4

    mask = fix(mask, levels)
    img_1 = fix(img_1, levels)
    img_2 = fix(img_2, levels)


    n_blend = img_1 * mask + (1 - mask) * img_2

    Gm = gaussianPyr(mask, levels)  # Build a Gaussian pyramid for mask

    La = laplaceianReduce(img_1, levels)  # Build a Laplacian pyramid for img_1
    Lb = laplaceianReduce(img_2, levels)  # Build a Laplacian pyramid for img_2

    Lc = La[levels - 1] * Gm[levels - 1] + (1 - Gm[levels - 1]) * Lb[levels - 1]

    for i in range(levels - 2, -1, -1):
        Lc = gaussExpand(Lc,gaussKernel) + La[i] * Gm[i] + (1 - Gm[i]) * Lb[i]

    return n_blend, Lc

    pass
