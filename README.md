# Image Sentiment Classification
This is an algorithm for classifying emotional images using CNN on keras and tensorflow framework.

## Data
Refer to 李宏毅's online courses homeworkd data. Each photo was specially processed to identify seven emotions.

training data (train.csv) : 22000 images

testing data (test.csv): 5500 images 

## Preprocess

Using 20 degrees of random rotation and turn around to improve the reliability of prediction

## CNN architecture:

In short: VGG-8 + BatchNormalization


The first six layers of convolution are repeated as follows. Filter size is 16, 32 and 64

Con2D (16 /BN/relu)

Con2D (16 /BN/relu)

MaxPooling2D ((2, 2))

Dropout (0.25)


Flatten () 

Dense(256/BN/relu) 

Dropout (0.5)


Dense(7/BN/softmax) 

## Result
Prediction accuracy: 67%
