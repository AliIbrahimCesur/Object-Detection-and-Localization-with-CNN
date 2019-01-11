# USAGE
# python classify.py --model fashion.model --labelbin mlb.pickle --image examples/8.jpeg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#from pyimagesearch.transform import four_point_transform
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

def classify(image):
	model = load_model("/home/ali/Desktop/keras-multi-dataset/keras/fashion.model")
	output = imutils.resize(image, width=400)
 
	# pre-process the image for classification
	image = cv2.resize(image, (125, 125))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# load the trained convolutional neural network and the multi-label
	# binarizer
	print("[INFO] loading network...")
	mlb = pickle.loads(open("/home/ali/Desktop/keras-multi-dataset/keras/mlb.pickle", "rb").read())

	# classify the input image then find the indexes of the two class
	# labels with the *largest* probability
	print("[INFO] classifying image...")
	proba = model.predict(image)[0]
	idxs = np.argsort(proba)[::-1][:1]


	# loop over the indexes of the high confidence class labels
	for (i, j) in enumerate(idxs):
		# build the label and draw the label on the image
		label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
		cv2.putText(output, label, (10, (i * 30) + 25), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

	# show the probabilities for each of the individual labels
	for (label, p) in zip(mlb.classes_, proba):
		print("{}: {:.2f}%".format(label, p * 100))

	# show the output image
	cv2.imshow("Output", output)

	return proba[0]*100

###--###
# load the image, convert it to grayscale, and blur it slightly


image = cv2.imread("/home/ali/Desktop/keras-multi-dataset/keras/examples/2.jpeg")
model = load_model("/home/ali/Desktop/keras-multi-dataset/keras/fashion.model")
output = imutils.resize(image, width=200)
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
#thresh = cv2.threshold(gray, 20, 150, cv2.THRESH_BINARY)[1]
#thresh = cv2.erode(thresh, None, iterations=2)
#thresh = cv2.dilate(thresh, None, iterations=2)

###--###



ar1=[]
temp=0
x_val=0
y_val=0
rat = classify(output)
print (rat)
NewImage = np.copy(output)
#cv2.imshow("Output3", NewImage)
height, width, color = NewImage.shape
print(height,width)

rat2=rat
output1 = np.copy(NewImage)
while rat2>rat-5:
	output1 = np.copy(NewImage)
	temp=rat2
	for i in range(y_val, (y_val+20)):
		for j in range(0, width):         
			NewImage[i,j] = 0
	rat2 = classify(NewImage)
	
	y_val+=20

rat3 = temp
rat4=rat3

print("alttan")
print(rat3)
output2 = np.copy(output1)

while rat4>rat3-5:
	output2 = np.copy(output1)
	temp = rat4
	for i in range((height-20-x_val), height-x_val):
		for j in range(0, width):         
			output1[i,j]=0
	rat4 = classify(output1)
	x_val+=20

x1_val = 0
y1_val = 0

rat5 = temp
rat6 = rat5
print("soldan")
print(rat5)
output3 = np.copy(output2)

while rat6>rat5-5:
	output3 = np.copy(output2)
	temp = rat6
	for j in range(y1_val, (y1_val+20)):
		for i in range(0, height):         
			output2[i,j] = 0
	rat6 = classify(output2)
	y1_val+=20

rat7 = temp
rat8 = rat7
print("sagdan")
print(rat7)
output4 = np.copy(output3)

while rat8>rat7-1:
	output4 = np.copy(output3)
	for j in range((width-20-x1_val), (width-x1_val)):
		for i in range(0, height):         
			output3[i,j] = 0
	rat8 = classify(output3)
	x1_val+=20


#cv2.imshow("Output", output)
cv2.imshow("Output4", output4)
cv2.waitKey(0)















