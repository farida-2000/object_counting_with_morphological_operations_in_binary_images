import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# Read image in grayscale
gray_im = cv.imread("Blood-cells_12.Red-blood-ce.jpg", cv.IMREAD_GRAYSCALE)

# Adjust the contrast using gamma correction (y = 1.2)

gamma_gray = np.array(255 * (gray_im / 255) ** 1.2 , dtype='uint8')
# Apply thresholding: gussian weighted sum of neighbours
thresh = cv.adaptiveThreshold(gamma_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 255, 19) #19:size of neighbourhood used for thresholding 
thresh = cv.bitwise_not(thresh)

# Erosion followed by dilation: I tried different patterns of kernel and used the best one in my code
kernel = np.ones((3,1), np.uint8) #15 by 15 3by1 15,1 15by 5
img_dilation = cv.dilate(thresh, kernel, iterations=1)
img_erode = cv.erode(img_dilation,kernel, iterations=1)
# clean potential noise after morphological operations
img_erode = cv.medianBlur(img_erode, 7)
plt.subplot(221)
plt.title('Dilatation + erosion')
plt.imshow(img_erode, cmap="gray", vmin=0, vmax=255)

# Labeling technique

ret, labels = cv.connectedComponents(img_erode)
# ret: total number of labels
#label: array where each pixel is assigned a label refrer to its connected component
label_hue = np.uint8(189 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
#creating a binary image where the pixels with a label_hue value of 0 are black and all other pixels are white to visualize contours labels
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

plt.subplot(222)
plt.title('Objects counted:'+ str(ret-1))
plt.imshow(labeled_img)
print('objects number is:', ret-1)
plt.show()
# calculate centers
image = cv.imread("Blood-cells_12.Red-blood-ce.jpg")
gray= cv.imread("Blood-cells_12.Red-blood-ce.jpg", cv.IMREAD_GRAYSCALE)
#gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray, (13,1), #(5,5) (7,1) (13,1)
					cv.BORDER_DEFAULT)
#binary image convertion
ret, thresh = cv.threshold(blur, 140, 255,  #140:179
						cv.THRESH_BINARY_INV)
cv.imwrite("thresh.png",thresh)
contours, hierarchies = cv.findContours(
	thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#creating blank image to show contours after
empty = np.zeros(thresh.shape[:2],
				dtype='uint8')
#using opencv function, we draw each contour on blank image
cv.drawContours(empty, contours, -1,
				(255, 0, 0), 1)
cv.imwrite("Contours.png", empty)
#iterate over countours in for loop and find centers using moments opencv function
for i in contours:
	moments_contours = cv.moments(i)
	if moments_contours['m00'] != 0:
		x_loc = int(moments_contours['m10']/moments_contours['m00'])
		y_loc = int(moments_contours['m01']/moments_contours['m00'])
		#set the text and circling of contours and centers 
		cv.drawContours(image, [i], -1, (0, 255, 0), 0) #10:0
		cv.circle(image, (x_loc, y_loc), 5, (0, 0, 255), -1) #3:5
		cv.putText(image, "center", (x_loc - 20, y_loc - 20),
				cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
	print(f"x: {x_loc} y: {y_loc}")
cv.imwrite("res.jpg", image)

