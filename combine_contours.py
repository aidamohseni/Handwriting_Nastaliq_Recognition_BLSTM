import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt

# image = cv2.imread('c1.jpg')
image = cv2.imread('shafiq.jpg')
# get dimensions of image
dm = image.shape
# height, width, number of channels in image
height = image.shape[0]
width = image.shape[1]
channels = image.shape[2]

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
"""plt.imshow(image)
plt.show()"""

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Find Canny edges
edged = cv2.Canny(gray, 30, 200)
"""plt.imshow(edged)
plt.show()"""

# Finding Contours
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
"""plt.imshow(edged)
plt.show()"""
# --------------------------------------------------
i = 0
boundingBoxes = [cv2.boundingRect(c) for c in contours]
(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][i], reverse=True))
# --------------------------------------------------
blank_image = 255 * np.ones_like(image, dtype=np.uint8)
print("Number of Contours found = " + str(len(contours)))
cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 3)
cv2.imwrite("shafiq/img.jpg", blank_image)
"""cv2.imshow('Contours', blank_image)
cv2.waitKey(0)"""

dimensions = list()
# for ct in rv_contours:
i = 0
for ct in contours:
    bimg = 255 * np.ones_like(image, dtype=np.uint8)
    cv2.drawContours(bimg, [ct], 0, (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y, w, h = cv2.boundingRect(ct)
    M = cv2.moments(ct)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    org = (cx, cy)
    cv2.putText(bimg, str(i), org, font, 1, (44, 11, 123), 2)
    cv2.imwrite("shafiq/img" + str(i) + ".jpg", bimg)
    # print([ct])
    cv2.imshow('Contours', bimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print("contour " + str(i), "= x,y,w,h", x, y, w, h, "\n")
    i += 1
# +++++++++++++++++++++++++++++++++++++++++++
# finding min and max horizontal coordination of a contour
# and save in an array as dimensions
for ct in contours:
    x, y, w, h = cv2.boundingRect(ct)
    print("min , max = ", x, x + w - 1, "========\n")
    dimensions.append([x, x + w - 1])
# ____________________________________________
for d in dimensions:
    i = dimensions.index(d)
    a = list()
    a.append(dimensions.index(d))
# ____________________________________________