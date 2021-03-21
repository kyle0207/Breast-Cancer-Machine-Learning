#%%
#importing necessary libraries and packages.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import callbacks
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd

"""IMPORTANT"""

#only for google colab GPU
# os.chdir('/content/drive/My Drive/Machine Learning/')
#for kyle
os.chdir('C:\\Users\\liewz\\Desktop\\Imperial\\Academic\\Year 2\\I-Explore\\breast-cancer-classification\\breast-cancer-classification')

#otherwise change this to whatever the base directory in File_Paths is.


import File_Paths
from Neural_Net import CancerNet
#%%
#initialising key parameters for training.
NUM_EPOCHS = 40 #number of epochs
INIT_LR = 1e-2 #initial learning rate
BS = 32 #batch size

augmentdata=1

base_dir=File_Paths.base_dir
#%%
try:
    os.mkdir(f'{base_dir}\\models')
except:
    None

#%%
#DATASET 1
# TRAIN_PATH = File_Paths.TRAIN_PATH_1 #insert training path
# TRAIN_PATH = f'{base_dir}//{TRAIN_PATH}'
# VAL_PATH = File_Paths.VAL_PATH_1 #insert validation path
# VAL_PATH = f'{base_dir}//{VAL_PATH}'

#DATASET 2
# TRAIN_PATH = File_Paths.TRAIN_PATH_2 #insert training path
# TRAIN_PATH = f'{base_dir}//{TRAIN_PATH}'
# VAL_PATH = File_Paths.VAL_PATH_2 #insert validation path
# VAL_PATH = f'{base_dir}//{VAL_PATH}'

#DATASET 3
# TRAIN_PATH = File_Paths.TRAIN_PATH_3 #insert training path
# TRAIN_PATH = f'{base_dir}//{TRAIN_PATH}'
# VAL_PATH = File_Paths.VAL_PATH_3 #insert validation path
# VAL_PATH = f'{base_dir}//{VAL_PATH}'

#DATASET 4
# TRAIN_PATH = File_Paths.TRAIN_PATH_4 #insert training path
# TRAIN_PATH = f'{base_dir}//{TRAIN_PATH}'
# VAL_PATH = File_Paths.VAL_PATH_4 #insert validation path
# VAL_PATH = f'{base_dir}//{VAL_PATH}'

#DATASET 5
# TRAIN_PATH = File_Paths.TRAIN_PATH_5 #insert training path
# TRAIN_PATH = f'{base_dir}//{TRAIN_PATH}'
# VAL_PATH = File_Paths.VAL_PATH_5 #insert validation path
# VAL_PATH = f'{base_dir}//{VAL_PATH}'

TRAIN_PATH = File_Paths.FULL_TRAIN#insert training path
TRAIN_PATH = f'{base_dir}//{TRAIN_PATH}'
VAL_PATH = File_Paths.TEST_PATH#insert validation path
VAL_PATH = f'{base_dir}//{VAL_PATH}'


TEST_PATH = File_Paths.TEST_PATH#insert testing path
TEST_PATH = f'{base_dir}//{TEST_PATH}'

# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(TRAIN_PATH)) # puts the image filenames into a list
totalTrain = len(trainPaths) # just the number of training data
totalVal = len(list(paths.list_images(VAL_PATH))) # just the number of validation data
totalTest = len(list(paths.list_images(TEST_PATH))) # just the number of testing data
#%%
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths] #making a list giving the score for each cell image
no_0 = trainLabels.count(0) #number of negatives in the training dataset
no_1 = trainLabels.count(1) #number of positives in the training dataset

#determining the weight of each class, i.e. 0 and 1, for future use.
#this is important because it takes into account the fact that we have an imbalance in class sizes.
classWeight = {0:max(no_0,no_1)/no_0,1:max(no_0,no_1)/no_1}
print('definitions done')
#%%
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
	TRAIN_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode="categorical",
	target_size=(48, 48),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

print("data augmented/sorted")

# include early stopping through callback
early_stopping = callbacks.EarlyStopping(
    min_delta=0.002,
    patience=10,
    restore_best_weights=True,
)

# initialize our CancerNet model and compile it
model = CancerNet.build(width=48, height=48, depth=3,
	classes=2)
opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS) #using Adagrad as the optimiser - stochastic gradient decay
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]) #keeping these as history that we will use to call to plot graphs

print('model built')
#%%
# fit the model
H = model.fit(
	x=trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	class_weight=classWeight,
	epochs=NUM_EPOCHS,
	callbacks=[early_stopping])

print('model learned')

model.save(f'{base_dir}\\models')
#%%
print('-'*15, 'evalution', '-'*15)

# use the trained model to make predictions on the test dataset
testGen.reset()
predIdxs = model.predict(x=testGen, steps=(totalTest // BS) + 1)

# find the index of the label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",  #name the plot here***
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# confusion matrix
cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))

# compute accuracy, sensitivity, and specificity
accuracy = (cm[0, 0] + cm[1, 1])/total
sensitivity = cm[0, 0]/(cm[0, 0] + cm[0, 1])
specificity = cm[1, 1]/(cm[1, 0] + cm[1, 1])

# print the results
print(cm)
print("acc: {:.4f}".format(accuracy))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# training loss and accuracy plot
H_history = pd.DataFrame(H.history)
print("Minimum Validation Loss: {:0.4f}".format(H_history['val_loss'].min()))
plt.style.use("ggplot")
H_history.loc[0:, ['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot()
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch')
plt.ylabel("Loss/Accuracy")
# plt.savefig(args["plot"])
