#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path

Path.cwd()


# In[ ]:


import os
os.chdir('../Downloads')
#change to downloads


# In[ ]:


datasets_dir = '../Documents/datasets'
os.mkdir(datasets_dir)
orig_dir = '../Documents/datasets/orig'
os.mkdir(orig_dir)


# In[ ]:


from zipfile import ZipFile
with ZipFile('./archive.zip', 'r') as zip_ref:
    zip_ref.extractall('../Documents/datasets/orig')


# In[ ]:


import os

#original files directory
orig_dataset_dir = "../Documents/datasets/orig"

# base directory
base_dir = "../Documents/datasets/idc"
os.mkdir(base_dir)

# training, validation and test directories
train_dir = os.path.sep.join([base_dir, "training"])
validation_dir = os.path.sep.join([base_dir, "validation"])
test_dir = os.path.sep.join([base_dir, "testing"])

#percentage to divide training from the dataset
train_split = 0.8

#percentage of the training to use as the validation set
validation_split = 0.1

#%%
# In[ ]:


get_ipython().system('pip install opencv-python')
#installing openCV, needed by imutils


# In[ ]:


get_ipython().system('pip install imutils')
#installing imutils package

#%%
# In[ ]:


from imutils import paths
import random, shutil, os

# taking the paths of the original dataset directory and randomly shuffling them
imagePaths = list(paths.list_images(orig_dataset_dir))
random.seed(77)
random.shuffle(imagePaths)

# using train_split defined above separate train and test sets
i = int(len(imagePaths) * train_split)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# separate a defined percentage (validation_split) of train set into validation
i = int(len(trainPaths) * validation_split)
validationPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# defining new datasets
datasets = [
    ("training", trainPaths, train_dir),
    ("validation", validationPaths, validation_dir),
    ("testing", testPaths, test_dir)
]


# In[ ]:


# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
    # printing the name of split being created
    print("[NOW] Building the '{}' split".format(dType))

    # if the output base output directory does not exist, create it
    if not os.path.exists(baseOutput):
        print("[NOW] Creating the '{}' directory".format(baseOutput))
        os.mkdir(baseOutput)

    # loop over the input image paths
    for inputPath in imagePaths:
        # extracting the class labels from the title and label
        filename = inputPath.split(os.path.sep)[-1]
        label = filename[-5:-4]

        # label directory
        labelPath = os.path.sep.join([baseOutput, label])

        # if the label output directory does not exist, create it
        if not os.path.exists(labelPath):
            print("[NOW] Creating the '{}' directory".format(labelPath))
            os.mkdir(labelPath)

        # construct the path to the destination image and then copy
        # the image itself
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)


# In[ ]:


print('Summary of the Cleaned dataset:')

print('Training: IDC Negative:', len(os.listdir('../Documents/datasets/idc/training/0')))
print('Training: IDC Positive:', len(os.listdir('../Documents/datasets/idc/training/1')))
print('Validation: IDC Negative:', len(os.listdir('../Documents/datasets/idc/validation/0')))
print('Validation: IDC Positive:', len(os.listdir('../Documents/datasets/idc/validation/1')))
print('Testing: IDC Negative:', len(os.listdir('../Documents/datasets/idc/testing/0')))
print('Testing: IDC Positive:', len(os.listdir('../Documents/datasets/idc/testing/1')))

