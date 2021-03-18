#!/usr/bin/env python
# coding: utf-8

# ## Integrating in the config file like the tutorial

# In[ ]:


from pathlib import Path
import os
os.chdir('C:\\Users\\liewz\\Desktop\\Imperial\\Academic\\Year 2\\I-Explore\\breast-cancer-classification\\breast-cancer-classification') 

from zipfile import ZipFile
from imutils import paths
import File_Paths
import random, shutil, os
from sklearn.model_selection import KFold
import numpy as np
# Path.cwd()


# In[ ]:



os.chdir('C:\\Users\\liewz\\Desktop\\Imperial\\Academic\\Year 2\\I-Explore\\breast-cancer-classification\\breast-cancer-classification') 
#this is where the zipfile of datasets should be
#change to downloads


# In[ ]:

base_dir=File_Paths.base_dir

datasets_dir = f'{base_dir}/datasets'
try:
    os.mkdir(datasets_dir)
except:
    None

orig_dir = f'{base_dir}/datasets/orig'
try:
    os.mkdir(orig_dir)
except:
    None


# In[ ]:



with ZipFile('./archive.zip', 'r') as zip_ref:
    zip_ref.extractall(f'{base_dir}/datasets/orig')


os.chdir(f'{base_dir}')
#change to downloads

#%%
#removing additional overlapping files
IDC_reg_filePath = "./datasets/orig/IDC_regular_ps50_idx5"
if os.path.exists(IDC_reg_filePath):
    shutil.rmtree(IDC_reg_filePath) 
#%%
# taking the paths of the original dataset directory and randomly shuffling them
patientPaths = (os.listdir(File_Paths.ORIG_INPUT_DATASET))
random.seed(77)
random.shuffle(patientPaths)

# using train_split defined above separate train and test sets
i = int(len(patientPaths) * File_Paths.TRAIN_SPLIT)
trainPaths = patientPaths[:i]
testPaths = patientPaths[i:]


kfold=KFold(5, True, 1)
split_indices=list(kfold.split(trainPaths))
train_val_splits=[[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]

for i in range(len(split_indices)):
    for j in range(len(split_indices[0])):
        for k in split_indices[i][j]:
            train_val_splits[i][j].append(trainPaths[k])



print('PatientPaths: ', patientPaths [:2:], 'Length:', len(patientPaths))
print('TrainPaths: ', trainPaths [:2:],'Length:', len(trainPaths))
print('TestPaths:', testPaths [:2:], 'Length:', len(testPaths))


#%%
traindir = [[],[],[],[],[]]
for i in range(len(train_val_splits)):
    for j in range(len(train_val_splits[i][0])):
        traindir[i].append(f"{base_dir}/datasets/orig/{train_val_splits[i][0][j]}")


imageTrainPaths=[[],[],[],[],[]]
for i in range(len(traindir)):
    for j in range(len(traindir[i])):
        imageTrainPaths[i] += list(paths.list_images(traindir[i][j]))


validationdir = [[],[],[],[],[]]
for i in range(len(train_val_splits)):
    for j in range(len(train_val_splits[i][1])):
        validationdir[i].append(f"{base_dir}/datasets/orig/{train_val_splits[i][1][j]}")


imageValidationPaths = [[],[],[],[],[]]
for i in range(len(validationdir)):
    for j in range(len(validationdir[i])):
        imageValidationPaths[i] += list(paths.list_images(validationdir[i][j]))

testdir = []
for i in range(len(testPaths)):
    testdir.append(f"{base_dir}/datasets/orig/{testPaths[i]}")

imageTestPaths = []
for i in range (len(testPaths)):
    imageTestPaths += list(paths.list_images(testdir[i]))

fulltraindir=[]
for i in trainPaths:
    fulltraindir.append(f"{base_dir}/datasets/orig/{i}")

fullimageTrainPaths=[]
for i in fulltraindir:
    fullimageTrainPaths += list(paths.list_images(i))
#%%
# defining new datasets
datasets = [
    ("training_1", imageTrainPaths[0], File_Paths.TRAIN_PATH_1),
    ("validation_1", imageValidationPaths[0], File_Paths.VAL_PATH_1),
    ("training_2", imageTrainPaths[1], File_Paths.TRAIN_PATH_2),
    ("validation_2", imageValidationPaths[1], File_Paths.VAL_PATH_2),
    ("training_3", imageTrainPaths[2], File_Paths.TRAIN_PATH_3),
    ("validation_3", imageValidationPaths[2], File_Paths.VAL_PATH_3),
    ("training_4", imageTrainPaths[3], File_Paths.TRAIN_PATH_4),
    ("validation_4", imageValidationPaths[3], File_Paths.VAL_PATH_4),
    ("training_5", imageTrainPaths[4], File_Paths.TRAIN_PATH_5),
    ("validation_5", imageValidationPaths[4], File_Paths.VAL_PATH_5),    
    
    ("full_training", fullimageTrainPaths, File_Paths.FULL_TRAIN),
    ("testing", imageTestPaths, File_Paths.TEST_PATH),   
]


#%%

for (dType, imagePaths, baseOutput) in datasets:
    # show which data split we are creating
    print("[INFO] building '{}' split".format(dType))

    # if the output base output directory does not exist, create it
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)

    # loop over the input image paths
    for inputPath in imagePaths:
        # extract the filename of the input image and extract the
        # class label ("0" for "negative" and "1" for "positive")
        filename = inputPath.split(os.path.sep)[-1]
        label = filename[-5:-4]

        # build the path to the label directory
        labelPath = os.path.sep.join([baseOutput, label])

        # if the label output directory does not exist, create it
        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)

        # construct the path to the destination image and then copy
        # the image itself
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)


# In[ ]:


print('[INFO] Summary of the Cleaned dataset:')

print('Training: IDC Negative:', len(os.listdir('./datasets/idc/training/0')))
print('Training: IDC Positive:', len(os.listdir('./datasets/idc/training/1')))
# print('Validation: IDC Negative:', len(os.listdir('./datasets/idc/validation/0')))
# print('Validation: IDC Positive:', len(os.listdir('./datasets/idc/validation/1')))
print('Testing: IDC Negative:', len(os.listdir('./datasets/idc/testing/0')))
print('Testing: IDC Positive:', len(os.listdir('./datasets/idc/testing/1')))

