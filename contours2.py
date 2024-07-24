import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt

# image = cv2.imread('c1.jpg')
image = cv2.imread('a77.jpg')
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
plt.imshow(edged)
plt.show()

blank_image = 255 * np.ones_like(image, dtype=np.uint8)
print("Number of Contours found = " + str(len(contours)))
cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', blank_image)
cv2.waitKey(0)

i = 0
rv_contours = reversed(contours)
dimensions = list()
for ct in rv_contours:
    """bimg = 255 * np.ones_like(image, dtype=np.uint8)
    cv2.drawContours(bimg, [ct], 0, (0, 255, 0), 2)
    cv2.imwrite("contours3/img" + str(i) + ".jpg", bimg)
    i += 1 
    #cv2.imshow('Contours', bimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()"""
    # print([ct])
    x, y, w, h = cv2.boundingRect(ct)
    print("min , max = ", x, x + w - 1, "========\n")
    dimensions.append([x, x + w - 1])
print(dimensions)
original_d = dimensions
assembly = list()
for d in dimensions:
    i = dimensions.index(d)
    a = list()
    a.append(dimensions.index(d))
    if d:
        print("d[0] =", d[0], "d[1]= ", d[1], "\n")
        for td in dimensions:
            if td:
                # print("td = ", td, "td[0] =", td[0], "td[1]= ", td[1], "\n")
                if (d[0] > td[0]) and (d[1] < td[1]):
                    ind = dimensions.index(td)
                    a.append(ind)
                    dimensions[ind] = None
                    print("ind =", ind)

    assembly.append(a)

print("----------------\n this is assembly\n")
print(assembly)
new_contour = list()
for c in assembly:
    bimg = 255 * np.ones_like(image, dtype=np.uint8)
    for b in c:
        cv2.drawContours(bimg, [contours[b]], 0, (0, 255, 0), 2)
        cv2.imwrite("contours4/img" + str(i) + ".jpg", bimg)
    i += 1
        # cv2.imshow('Contours', bimg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()