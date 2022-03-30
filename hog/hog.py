from sklearn.svm import LinearSVC
from skimage import exposure
from skimage import feature
import argparse
import cv2
import imutils
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True)
ap.add_argument("-t", "--test", required=True)

args = vars(ap.parse_args())

print("Extracting features...")
data = []
labels = []
for imagePath in paths.list_images(args["training"]):
    brand = imagePath[imagePath.rfind("\\") + 1:].split("_")[0]
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    c = max(cnts, key=cv2.contourArea)

    (x, y, w, h) = cv2.boundingRect(c)
    logo = gray[y:y + h, x:x + w]
    logo = cv2.resize(logo, (200, 100))

    H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True,
                    block_norm="L1")

    data.append(H)
    labels.append(brand)

print("Training classifier...")

model = LinearSVC()

model.fit(data, labels)
print("Evaluting...")

for (i, imagePath) in enumerate(paths.list_images(args["test"])):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logo = cv2.resize(gray, (200, 100))

    H, hogImage = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                              transform_sqrt=True, block_norm="L1", visualize=True)

    pred = model.predict(H.reshape(1, -1))[0]

    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    cv2.imshow("HOG image #{}".format(i + 1), hogImage)
    cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.imshow("Test image #{}".format(i + 1), image)
    cv2.waitKey(0)
