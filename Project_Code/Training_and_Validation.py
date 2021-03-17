#%%
#importing necessary libraries and packages.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import callbacks
from pyimagesearch.cancernet import CancerNet
from pyimagesearch import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

#%%
#initialising key parameters for training.
NUM_EPOCHS = 40 #number of epochs
INIT_LR = 1e-2 #initial learning rate
BS = 32 #batch size

augmentdata=1


#%%
TRAIN_PATH = config.TRAIN_PATH#insert training path
VAL_PATH = config.VAL_PATH#insert validation path
TEST_PATH = config.TEST_PATH#insert testing path

# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(TRAIN_PATH)) # puts the image filenames into a list
totalTrain = len(trainPaths) # just the number of training data
totalVal = len(list(paths.list_images(VAL_PATH))) # just the number of validation data
totalTest = len(list(paths.list_images(TEST_PATH))) # just the number of testing data

trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths] #making a list giving the score for each cell image
no_0 = trainLabels.count(0) #number of negatives in the training dataset
no_1 = trainLabels.count(1) #number of positives in the training dataset

#determining the weight of each class, i.e. 0 and 1, for future use.
#this is important because it takes into account the fact that we have an imbalance in class sizes.
classWeight = [max(no_0,no_1)/no_0,max(no_0,no_1)/no_1] 

# initialize the training data augmentation object
# This is basically the blueprint that allows for the augmentation of training data to be done
if augmentdata:
    trainAug = ImageDataGenerator(
    	rescale=1 / 255.0,
    	rotation_range=20,
    	zoom_range=0.05,
    	width_shift_range=0.1,
    	height_shift_range=0.1,
    	shear_range=0.05,
    	horizontal_flip=True,
    	vertical_flip=True,
    	fill_mode="nearest")
else:
    trainAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the validation (and testing) data augmentation object
# same thing, but for validation and testing data
valAug = ImageDataGenerator(rescale=1 / 255.0)

# note that both of these are scaled to 0-1


# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# include early stopping through callback
early_stopping = callbacks.EarlyStopping(
    min_delta=0.002,
    patience=15,
    restore_best_weights=True,
)

# initialize our CancerNet model and compile it
model = CancerNet.build(width=48, height=48, depth=3,
	classes=2)
opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS) #using Adagrad as the optimiser - stochastic gradient decay
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) #keeping these as history that we will use to call to plot graphs

# fit the model
H = model.fit(
	x=trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	class_weight=classWeight,
	epochs=NUM_EPOCHS,
        callbacks=[early_stopping])

# model = CancerNet.build(width=48, height=48, depth=3,
# 	classes=2)
# opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
# model.compile(loss="binary_crossentropy", optimizer=opt,
# 	metrics=["accuracy"])

# fit the model
# H = model.fit(
#	x=trainGen,
#	steps_per_epoch=totalTrain // BS,
#	validation_data=valGen,
#	validation_steps=totalVal // BS,
#	class_weight=classWeight,
#	epochs=NUM_EPOCHS)
