import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Input-Set/Cracked_01.jpg')
# img = cv2.imread('tree260/image/6192.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.blur(gray, (2, 2))

img_log = (np.log(blur + 1) / (np.log(1 + np.max(blur)))) * 255

img_log = np.array(img_log, dtype=np.uint8)

bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

edges = cv2.Canny(bilateral, 100, 200)

kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

sift = cv2.xfeatures2d.SIFT_create()

# orb = cv2.ORB_create(nfeatures=1500)
keypoints, descriptors = sift.detectAndCompute(closing, None)
featuredImg = cv2.drawKeypoints(closing, keypoints, None)

plt.subplot(121), plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(featuredImg, cmap='gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.show()
