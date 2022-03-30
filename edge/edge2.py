import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('tree260/image/6192.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

f, axes = plt.subplots(1,3, figsize=(30,10))
axes[0].imshow(img)
axes[0].set_title('origin')
axes[1].imshow(sobel_x)
axes[1].set_title('sobel_x')
axes[2].imshow(sobel_y)
axes[2].set_title('sobel_y')
plt.show()