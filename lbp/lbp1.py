import math
import cv2
import numpy as np
from skimage.feature import local_binary_pattern  # # pip install scikit-image
import matplotlib.pyplot as plt

KERNEL_WIDTH = 7
KERNEL_HEIGHT = 7
SIGMA_X = 3
SIGMA_Y = 3


def main():
    # img = cv2.imread('aaaa.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.imread('tree260/image/6192.jpg', cv2.IMREAD_GRAYSCALE)

    # LBP
    out = local_binary_pattern(image=img, P=8, R=1, method='default')
    cv2.imwrite('lbp.jpg', out)
    print("Saved image @ lbp.jpg")

    # Gaussian blur + LBP
    blur_img = cv2.GaussianBlur(img, ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)
    blur_out = local_binary_pattern(image=blur_img, P=8, R=1, method='default')

    f, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(blur_img)
    axes[0].set_title('origin')
    axes[1].imshow(blur_out)
    axes[1].set_title('result')
    plt.show()
    # cv2.imwrite('blur.jpg', blur_img)
    # cv2.imwrite('blur_lbp.jpg', blur_out)
    print("Saved image @ blur.jpg")
    print("Saved image @ blur_lbp.jpg")


if __name__ == "__main__":
    main()