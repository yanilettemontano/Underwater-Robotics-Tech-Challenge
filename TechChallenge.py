import cv2 #for image processing
import numpy as np #represent images as arrays
import matplotlib.pyplot as plt #display images


#testing python environment
#print(cv2.__version__)
#print(np.__version__)

#---------Importing image-----------
#image path
imagePath = 'underwater.jpg'
#load image 
image = cv2.imread(imagePath);

#Display original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #displays image and BGR --> RBG for proper display
plt.axis('off')
plt.show()

#--------Convert image BGR -> HSV-------
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #converts the image
#Displays the HSV image
plt.imshow(hsv)
plt.axis('off')
plt.show()

#--------Create Mask and color range-----
#Orange in OpenCV wraps around 0-20
#will need 2 ranges to combine
leftOrange1 = np.array([0, 100, 80]) 
rightOrange1 = np.array([20, 255, 255])

mask = cv2.inRange(hsv, leftOrange1, rightOrange1)

#Display mask
plt.imshow(mask, cmap='gray') #displays the mask in greyscale
plt.axis('off')
plt.show()

#-------Reducing Noise with morphological ops--------
#kernel for morphological operations
kernel = np.ones((4,4), np.uint8)
#erode removes all noise specks
mask = cv2.erode(mask, kernel, iterations=1)
#dilate restores the main object after erosion
mask = cv2.dilate(mask, kernel, iterations=2)
#------Find contours & draw bounding circles---------
#finds the contours (outlines of the orange region)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
output = image.copy()

cv2.drawContours(output, contours, -1, (255, 0, 0), 3)

for contour in contours:
    #the area threshold is 80 px squared is intentionally small here since the distant fish are tiny
    if cv2.contourArea(contour) < 80: continue
    #gets the bounding circle for each contour
    (x,y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    #draws the bounding circle and filters small areas for noise
    if radius > 10:
        cv2.circle(output, center, int(radius), (0, 255, 0), 2)

#display image with contours and bounding circles
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
