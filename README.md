# Methods for Interpreting Machine Learning Techniques in Blood Smear Image Based Leukaemia Classification

This code repository is part of a research article which explores the implementation of machine learning models using blood smear images to enhance diagnostic accuracy of leukaemia.

The full paper is included in the repository in the "Research_Article" directory.


## Requirements to run
This project was built with Python 3.11 along with the following packages:  
grad-cam-1.4.8  
lime-0.2.0.1  
matplotlib-3.8.2  
numpy-1.24.1  
scikit-learn-1.3.2  
seaborn-0.13.0  
shap-0.43.0  
torch-2.1.1  
torchsampler-0.1.2  
torchvision-0.16.1  


## Datasets
This project was made possible with two datasets:
1. ALL Challenge dataset of ISBI 2019 (C-NMC 2019)

This is a publicly available dataset provided by _The Cancer Imaging Archive_ [https://doi.org/10.7937/tcia.2019.dc64i46r](https://doi.org/10.7937/tcia.2019.dc64i46r).
Users are subject to the license and restrictions stated under _Citations & Data Usage Policy_ in the linked archive.

This dataset is used primarily to train the model however a portion of the dataset is split for validation (16%) and testing (20%).

2. ALL_IDB1 

This dataset requires special permission which can only be granted by the organisation.
See [https://scotti.di.unimi.it/all/](https://scotti.di.unimi.it/all/) for more details.

This dataset is used exclusively for testing of the model.

If you are unable to obtain the second dataset, the model can still be trained and evaluated.
just ignore _prepare_dataset2.py_ and _test_dataset2.py_ which are the only scripts that use it for further evaluation purposes


## Running
1. prepare_dataset1.py
- Run this after _C-NMC_Leukemia.zip_ has been placed inside the _Datasets_ folder
- Creates a new subfolder called _dataset1_ from the train and preliminary test compositions
- The final test set composition is not used since the ground truth is not provided with the images

2. train_model.py
- Defines the model architecture and trains it on the prepared dataset1
- Saves validation accuracy and validation loss in _Cache/training_valloss_valaccuracy.json_ for _visualise_training.py_
- Saves _Cache/dataset1_mean+std.pth_ and _Cache/dataset1_test_dataloader.pth_ for _test_dataset1.py_
- Saves model weights to _Model_Weights/vgg19.pth_

3. visualise_training.py
- Generates a graph of the validation accuracy and validation loss from the values generated in _train_model.py_
- This figure is saved in _Metrics/Accuracy and Loss vs epoch.png_

3. test_dataset1.py
- Outputs several key metrics including accuracy, precision, recall and F1 score.
- Generates a confusion matrix at _Metrics/dataset1 - Confusion Matrix.png_
- Generates 2 figures for each interpretability model (Grad-Cam++, LIME and SHAP); one for 3 correctly predicted images and one for 3 incorrectly predicted images

5. prepare_dataset2.py
- Run this after the _ALL_IDB1_ folder has been placed inside the _Datasets_ folder
- Creates sliding windows from each image and labels them cancerous and noncancerous depending if the centroid of the any cancerous cells are present in the image

6. test_dataset2.py
- This does the same as _test_dataset1.py_ but using _dataset2_
- Each image is named with _dataset2_ included so it will not overwrite any images generated from _test_dataset1.py_
