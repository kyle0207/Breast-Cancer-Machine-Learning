# Breast Cancer Machine Learning Project
This GitHub Repository is for the Interdisciplinary Research Computing **Group 2**. The repository includes information relevant to the project. Each of the python files included in the repository are also explained below.

# Research Question
> #### How can we use a computer algorithm to match different breast cancer images with their corresponding diagnosis? 
>

# Instructions for use

**Adding base directories**:
This **has** to be done in File_Paths.py, Final_Cleaning_Dataset.py, and Training_and_Validation.py. The exact lines will be under the line titled **"""IMPORTANT"""**. 

	- File_Paths.py:
		- Change the variable base_dir to the project directory.
	- Final_Cleaning_Dataset.py and Training_and_Validation.py:
		- Change the argument of os.chdir() contained within the import library cell to the project directory as desired.

**Downloading dataset**:
The dataset can be downloaded at "https://www.kaggle.com/paultimothymooney/breast-histopathology-images/download". The file will be downloaded as a zip file and it should be placed within the base directory mentioned above. The codes should take care of sorting the data out into appropriate subdirectories. **Do not decompress the zipfile before running the codes.**

**Order of running code**:
The order is as follows:
1. File_Paths.py
2. Final_Cleaning_Dataset.py
3. Training_and_Validation.py
4. cnn_visualization.py *(optional)*

# Information for each of the python modules

* **Neural_Net.py**:
This provides the architecture on which the neural network uses. This is taken from the tutorial at "https://www.pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/"

* **File_Paths.py**: 
Contains all the file paths necessary for the project

* **Final_Cleaning_Dataset.py**:
Re-organizes and cleans the Kaggle dataset (https://www.kaggle.com/paultimothymooney/breast-histopathology-images) after unpacking it from the zip file. This code will make sure that a directory is made for each dataset that will be used in a 5-fold cross validation (k=5) process. The images are randomly shuffled by patient ID (to avoid biological bias) into training and testing sets. Adapted from the tutorial mentioned above. 'Project_Cleaning_Datasets.py' (in the 'Non-essentials folder') is a older version of the code that randomly separates the same dataset into train, validation and testing (i.e. not for cross-validation).

* **Training_and_Validation.py**:
The code that does the deep learning of the data. It also gives the performance metrics for each trained model.

* **cnn_visualization.py**:
The code to create the flowchart of the layers in the CNN and to visualize each layer using a sample image that was not used for training the model. Visualization of the sample image was represented as a 4D tensor with color-coding. The codes used here were created following the codes described in the book: "Chollet, F. (2018) Deep learning with Python. Shelter Island, NY, Manning. Available from: http://media.obvsg.at/AC13785830-4001 ."