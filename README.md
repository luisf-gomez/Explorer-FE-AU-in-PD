# Explorer-FE-AU-in-PD
This GitHub provides the source code for the paper "Exploring Facial Expression and Action Units in Parkinson Disease"

## Support information
### Computer technical characteristics

- **O.S:** Windows 10

- **Processor:** Intel Core i7 - 7800x

- **GPU:** NVIDIA GeForce GTX 1080TI 

- **Enviroment:** Python 3.8.5 with TensorFlow 2.3.0 

### Training hyperparameters

- **Regularization:** Dropout (0.5 probability) and Early Stopping (10 epoch)

- **Batch size:** 128 Frames (Train with EmotionNet) and 16 frames (Train with FacePark-GITA)

- **Optimizer:** ADAM (Train with EmotionNet) and SGD (Train with FacePark-GITA)

### **FAU Training time:**
Training times using the Emotionet database took about 28.6 hours and 23.4 hours for models based on ResNet50 architectures and models from scratch, respectively. 

### **From Scratch training time:**
It took approximately around 30 minutes for each model to train from scratch with the FacePark-GITA database.

### **Features evaluation on SVMs:**
The grid search for the Support Vector Machines (SVMs) took about 18 minutes for each model based on ResNet50 and 8 minutes for each model from scratch.

## Prerequisites

- Python 3.8.5 or higher: Preferably a new environment
- All the system was probed on Tensorflow=2.3.0 
- All the datasets and labels used in this work cannot be shared 


## Run information

- Train the AU models with the scripts in **/Models**:
	- **Model_Freeze.py**
	- **Model_VGG8.py**
	- **Model_ResNet7.py**
	
- Use in **/Models/Models_TripleLoss.py** to create and extract the embeddings related to AU and PD.
- Use in **/Features/FA_Features.py** to extract the embeddings related to FA.

- Inside all the **/Features/\*/\*/** folders, you can run **Optimization_SVM_\*.py** files to train models.

- **/Features/BestParam.py** check the models trained and inform the best hyperparameters to the SVM-linear and SVM-rbf

- **/Features/25RandCV.py** Run 25 CrossValidation Subject-Independent to get a big group of samples to made statistical tests

- **/Features/StatisticalTest/RunStatisticalTests.py** run the nonparametrics Kruskall-Wallis test and Mann-Whitney U test with the 25 random CV results
