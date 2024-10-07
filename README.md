# CO-ResNetRS50-SSL
A semi-supervised image classification method based on an improved ResNetRS50 architecture, termed CO-ResNetRS50-SSL.
# Usage Example
First, clone this project using Git.
### 1. Download the Dataset
Go to (link) to download the full version of the dataset, then place all images into the RiceImagesData folder under the data directory.
### 2. Set Up the Environment
Run 'pip intall -r requirements.txt'.
### 3. Navigate to the Directory
Change your path to the src directory.
### 4. Training
Run ResNetRS50_train.py to train the ResNetRS50 network. The trained model files will be saved in the ResNetRS50 folder under the model directory, and the training and testing result data will be saved in the ResNetRS50 folder under the data directory.

Run ResNetRS50_SSL_train.py to train the ResNetRS50-SSL network. You can set the confidence threshold parameter with confidence-threshold, which defaults to 0.90. By adjusting the save path in the code, you can save the trained model files and the training process data in the corresponding directory. The same applies to C-ResNetRS50-SSL and CO-ResNetRS50-SSL.
### 5. Testing
Run test.py. By setting different model parameter paths and calling different models, you can obtain the results of the models on the test set.
### 6. Plotting
Run fig.py. By setting the method names in main(), you can generate corresponding plots from the saved data files.
