# Explorer-FE-AU-in-PD
This GitHub provides the source code for the paper "Exploring Facial Expression and Action Units in Parkinson Disease"

## Prerequisites

- Python 3.8.5 or higher: Preferably a new environment
- All the system was probed on Tensorflow=2.3.0 and a NVIDIA GeForce GTX 1080 TI
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
