# import the necessary packages
import os


#for google colab GPU
# base_dir='/content/drive/My Drive/Machine Learning/'
#for kyle
# base_dir='C:\\Users\\liewz\\Desktop\\Imperial\\Academic\\Year 2\\I-Explore\\breast-cancer-classification\\breast-cancer-classification'
"""IMPORTANT"""

base_dir= #set your own directory here where all everything will go

os.chdir(base_dir)
# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "datasets//orig"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "datasets//CVdatasets"
try:
    os.mkdir(BASE_PATH)
except:
    None

DATASET_1=f"{BASE_PATH}//dataset1"
try:
    os.mkdir(DATASET_1)
except:
    None
TRAIN_PATH_1 = f"{DATASET_1}//training"
VAL_PATH_1 = f"{DATASET_1}//validation"
try:
    os.mkdir(TRAIN_PATH_1)
except:
    None
try:
    os.mkdir(VAL_PATH_1)
except:
    None


DATASET_2=f"{BASE_PATH}//dataset2"
try:
    os.mkdir(DATASET_2)
except:
    None
TRAIN_PATH_2 = f"{DATASET_2}//training"
VAL_PATH_2 = f"{DATASET_2}//validation"
try:
    os.mkdir(TRAIN_PATH_2)
except:
    None
try:
    os.mkdir(VAL_PATH_2)
except:
    None
    
DATASET_3=f"{BASE_PATH}//dataset3"
try:
    os.mkdir(DATASET_3)
except:
    None
TRAIN_PATH_3 = f"{DATASET_3}//training"
VAL_PATH_3 = f"{DATASET_3}//validation"

try:
    os.mkdir(TRAIN_PATH_3)
except:
    None
try:
    os.mkdir(VAL_PATH_3)
except:
    None

DATASET_4=f"{BASE_PATH}//dataset4"
try:
    os.mkdir(DATASET_4)
except:
    None
TRAIN_PATH_4 = f"{DATASET_4}//training"
VAL_PATH_4 = f"{DATASET_4}//validation"

try:
    os.mkdir(TRAIN_PATH_4)
except:
    None
try:
    os.mkdir(VAL_PATH_4)
except:
    None
    
DATASET_5=f"{BASE_PATH}//dataset5"
try:
    os.mkdir(DATASET_5)
except:
    None
TRAIN_PATH_5 = f"{DATASET_5}//training"
VAL_PATH_5 = f"{DATASET_5}//validation"

try:
    os.mkdir(TRAIN_PATH_5)
except:
    None
try:
    os.mkdir(VAL_PATH_5)
except:
    None
#%%
# derive the training, validation, and testing directories
    
TEST_PATH = f"{BASE_PATH}//testing"
try:
    os.mkdir(TEST_PATH)
except:
    None
    
FULL_TRAIN = f"{BASE_PATH}//full_training_dataset"
try:
    os.mkdir(FULL_TRAIN)
except:
    None
# define the amount of data that will be used training
TRAIN_SPLIT = 0.8
