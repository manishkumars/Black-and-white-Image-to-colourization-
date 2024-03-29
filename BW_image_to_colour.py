import numpy as np
import matplotlib.pyplot as plt
import cv2

#Define Image to be colorized:

IMAGE = "BW.jpg"

#Define Model Paths:

prototxt = "./model/colorization_deploy_v2.prototxt"
model = "./model/colorization_release_v2.caffemodel"
points = "./model/pts_in_hull.npy"
image =  "./input_images/"+IMAGE

#Load serialized black and white colorizer model and cluster:

net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)

#Add the cluster centers as 1x1 convolutions to the model:

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

#Load the input image, scale it and convert it to Lab:

image = cv2.imread(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# using matplotlib

plt.imshow(image)
plt.axis('off');

#Extracting “L”:

scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

#Predicting “a” and “b”:

net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

#Creating a colorized Lab photo (L + a + b):

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

#colourized lab image

plt.imshow(colorized)
plt.axis('off');
plt.title('colorized LAB image');

#Converting it to RGB:

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")
plt.imshow(colorized)
plt.axis('off');

#Saving the final RGB photo:

cv2.imwrite("./colorized_images/Color_"+IMAGE, cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
