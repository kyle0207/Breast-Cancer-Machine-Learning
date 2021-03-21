# Breast Cancer Machine Learning Project
This GitHub Repository is for the Interdisciplinary Research Computing Group 2. The repository includes the information relevant to the project.

# Research Question
How can we use a computer algorithm to match different breast cancer images with their corresponding diagnosis? 

# Instructions for use

Adding base directories:
This has to be done in File_Paths.py, Final_Cleaning_Dataset.py, and Training_and_Validation.py. The exact lines will be under the line titled """IMPORTANT""". 
	- File_Paths.py:
		- Change the variable base_dir to the project directory.
	- Final_Cleaning_Dataset.py and Training_and_Validation.py:
		- Change the argument of os.chdir() contained within the import library cell to the project directory 		  desired.

Downloading dataset:
The dataset can be downloaded at "https://www.kaggle.com/paultimothymooney/breast-histopathology-images/download". This will be downloaded as a zip file, and it should be placed within the base directory mentioned above. The codes should take care of sorting the data out.

Order of running code:
The order is as follows:
-File_Paths.py
-Final_Cleaning_Dataset.py
-Training_and_Validation.py

# Information for each of the python modules

-Neural_Net.py
This provides the architecture on which the neural network uses. This is taken from the tutorial at "https://www.pyimagesearch.com/2019/02/18/breast-cancer-classification-with-keras-and-deep-learning/"

-File_Paths.py
Contains all the file paths necessary for the project

-Final_Cleaning_Dataset.py
Reorganises and cleans the code after unpacking it from the zip file. This code will make sure that a directory is made for each dataset that will be used in a 5 fold cross validation process. Adapted from the tutorial mentioned above.

-Training_and_Validation.py
The code that does the deep learning of the data. It also gives the performance metrics for each trained model.