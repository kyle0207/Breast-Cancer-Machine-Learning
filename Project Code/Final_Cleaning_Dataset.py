#!/usr/bin/env python
# coding: utf-8

# ## Integrating in the config file like the tutorial

# In[ ]:


from pathlib import Path

Path.cwd()


# In[ ]:


import os
os.chdir('../Downloads') #this is where the zipfile of datasets should be
#change to downloads


# In[ ]:

base_dir='../Documents'

datasets_dir = f'{base_dir}/datasets'
os.mkdir(datasets_dir)
orig_dir = f'{base_dir}/datasets/orig'
os.mkdir(orig_dir)


# In[ ]:


from zipfile import ZipFile
with ZipFile('./archive.zip', 'r') as zip_ref:
    zip_ref.extractall(f'{base_dir}/datasets/orig')


# In[ ]:


get_ipython().system('pip install opencv-python')
#installing openCV, needed by imutils


# In[ ]:


get_ipython().system('pip install imutils')
#installing imutils package


# In[ ]:


import os
os.chdir(f'{base_dir}')
#change to downloads


# In[ ]:


from imutils import paths
import File_Paths
import random, shutil, os

#removing additional overlapping files
IDC_reg_filePath = "./datasets/orig/IDC_regular_ps50_idx5"
if os.path.exists(IDC_reg_filePath):
    shutil.rmtree(IDC_reg_filePath) 

# taking the paths of the original dataset directory and randomly shuffling them
patientPaths = (os.listdir(File_Paths.ORIG_INPUT_DATASET))
random.seed(77)
random.shuffle(patientPaths)

# using train_split defined above separate train and test sets
i = int(len(patientPaths) * File_Paths.TRAIN_SPLIT)
trainPaths = patientPaths[:i]
testPaths = patientPaths[i:]

# separate a defined percentage (validation_split) of train set into validation
# i = int(len(trainPaths) * File_Paths.VAL_SPLIT)
# validationPaths = trainPaths[:i]
# trainPaths = trainPaths[i:]

# defining new datasets
datasets = [
    ("training", trainPaths, File_Paths.TRAIN_PATH),
    # ("validation", validationPaths, File_Paths.VAL_PATH),
    ("testing", testPaths, File_Paths.TEST_PATH)
]


# In[ ]:


imagePaths = list(paths.list_images(orig_dir))

print('PatientPaths: ', patientPaths [:2:], 'Length:', len(patientPaths))
print('TrainPaths: ', trainPaths [:2:],'Length:', len(trainPaths))
print('TestPaths:', testPaths [:2:], 'Length:', len(testPaths))

print('imagePaths: ',imagePaths [:2:])


# In[ ]:


traindir = []
for i in range(len(trainPaths)):
    traindir.append(f"{base_dir}/datasets/orig/{trainPaths[i]}")
print(traindir)


# In[ ]:


imageTrainPaths = []
for i in range (len(trainPaths)):
    imageTrainPaths += (paths.list_images(traindir[i]))


# In[ ]:


testdir = []
for i in range(len(testPaths)):
    testdir.append(f"{base_dir}/datasets/orig/{testPaths[i]}")

imageTestPaths = []
for i in range (len(testPaths)):
    imageTestPaths += (paths.list_images(testdir[i]))


# In[ ]:


# validationdir = []
# for i in range(len(validationPaths)):
#     validationdir.append(f"../Documents/datasets/orig/{validationPaths[i]}")

# imageValidationPaths = []
# for i in range (len(validationPaths)):
#     imageValidationPaths += (paths.list_images(validationdir[i]))


# In[ ]:


for (dType, imagePaths, baseOutput) in datasets:
    # show which data split we are creating
    print("[INFO] building '{}' split".format(dType))

    # if the output base output directory does not exist, create it
    if not os.path.exists(baseOutput):
        print("[INFO] 'creating {}' directory".format(baseOutput))
        os.makedirs(baseOutput)

    # loop over the input image paths
    for inputPath in imageTrainPaths:
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

    for inputPath in imageTestPaths:
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
        
    # for inputPath in imageValidationPaths:
    #     # extract the filename of the input image and extract the
    #     # class label ("0" for "negative" and "1" for "positive")
    #     filename = inputPath.split(os.path.sep)[-1]
    #     label = filename[-5:-4]

    #     # build the path to the label directory
    #     labelPath = os.path.sep.join([baseOutput, label])

    #     # if the label output directory does not exist, create it
    #     if not os.path.exists(labelPath):
    #         print("[INFO] 'creating {}' directory".format(labelPath))
    #         os.makedirs(labelPath)

    #     # construct the path to the destination image and then copy
    #     # the image itself
    #     p = os.path.sep.join([labelPath, filename])
    #     shutil.copy2(inputPath, p)


# In[ ]:


print('[INFO] Summary of the Cleaned dataset:')

print('Training: IDC Negative:', len(os.listdir('./datasets/idc/training/0')))
print('Training: IDC Positive:', len(os.listdir('./datasets/idc/training/1')))
# print('Validation: IDC Negative:', len(os.listdir('./datasets/idc/validation/0')))
# print('Validation: IDC Positive:', len(os.listdir('./datasets/idc/validation/1')))
print('Testing: IDC Negative:', len(os.listdir('./datasets/idc/testing/0')))
print('Testing: IDC Positive:', len(os.listdir('./datasets/idc/testing/1')))

