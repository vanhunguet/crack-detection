import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt

KERNEL_WIDTH = 7
KERNEL_HEIGHT = 7
SIGMA_X = 3
SIGMA_Y = 3


def main():
    # img = cv2.imread('tree260/image/6192.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('tree260/image/6192.jpg')
    # LBP
    # out = local_binary_pattern(image=img, P=8, R=1, method='default')
    # cv2.imwrite('lbp.jpg', out)
    # print("Saved image @ lbp.jpg")

    img = cv2.imread('Input-Set/Cracked_01.jpg')

    # Convert into gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Image processing ( smoothing )
    # Averaging
    # blur = cv2.blur(gray, (3, 3))

    # Gaussian blur + LBP
    blur_img = cv2.GaussianBlur(gray, ksize=(KERNEL_WIDTH, KERNEL_HEIGHT), sigmaX=SIGMA_X, sigmaY=SIGMA_Y)
    blur_out = local_binary_pattern(image=blur_img, P=8, R=1, method='default')
    img_log = (np.log(blur_out + 1) / (np.log(1 + np.max(blur_out)))) * 255

    # Specify the data type
    img_log = np.array(img_log, dtype=np.uint8)

    # Image smoothing: bilateral filter
    bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

    # Canny Edge Detection
    edges = cv2.Canny(bilateral, 100, 200)

    # Morphological Closing Operator
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Create feature detecting method
    # sift = cv2.xfeatures2d.SIFT_create()
    # surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=1500)

    # Make featured Image
    keypoints, descriptors = orb.detectAndCompute(closing, None)
    featuredImg = cv2.drawKeypoints(closing, keypoints, None)

    plt.subplot(121),plt.imshow(img)
    plt.title('Original'),plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(featuredImg,cmap='gray')
    plt.title('Output Image'),plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()