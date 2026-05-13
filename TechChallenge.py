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
plt.figure(figsize=(6,6)) # 6 x 6 inches
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) #displays image and BGR --> RBG for proper display
plt.axis('off')
plt.show()

#--------Convert image BGR -> HSV-------
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #converts the image
#Displays the HSV image
plt.figure(figsize=(6,6))
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
plt.figure(figsize=(6,6))
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()