# Animal Classification Application

## 1.	Problem Statement and Introduction:

Efficient and reliable monitoring of wild animals in their Natural habitats is essential to inform conservation and management decisions regarding wildlife species, migration patterns, and habitat protection, and is possible, rehabilitation and grouping species of the same animals together. Processing a large volume of images and videos captured from camera traps manually is extremely expensive, time consuming, and monotonous. This presents a major obstacle to scientists and ecologists to monitor wildlife in an open environment.

Processing a large volume of images and videos captured from camera traps manually is extremely expensive, time-consuming, and monotonous. This presents a major obstacle to scientists and ecologists to monitor wildlife in an open environment. Images captured in a field represent a challenging task while classifying since they appear in a different pose, cluttered background, different lighting and climatic conditions, human photographic errors, different angles, and occlusions.


## 2.	Dataset Selection selection 

There are many datasets available for the animal classification datasets among that we selected below 3 datasets with different classes and different number of images per class.

| Info 		         |Dataset 1|Dataset 2|Dataset 3|
| -----------------------|---------|---------|---------|
| Original  Categories   | 2	  |25       | 12       |		
| Original No. of Images | 25k    | 15.1K   | 17.2K   |
| Modified Categories.   | 2| 5| 12| 
| Modified No. Categories| 5k| 10K| 15K|

The first dataset, called Dataset 1, contains approximately 25,000 RGB images of dogs and cats with varying pixel resolutions, ranging from 320 x 200 to 498 x 480 pixels.  Originally created by Microsoft Research for a Kaggle competition in 2013, we modified this dataset and selected 5,000 images for our training. Images within this dataset are in .jpg format.

The second dataset, Dataset 2, was provided by a user named Saumil Agrawal in 2018. This dataset contains around 15,100 images of 25 different animal classes, with a fixed pixel resolution of 1280 x 720 pixels. We observed that this dataset was highly imbalanced, with some classes having only 60 images. To avoid potential bias towards certain classes, we selected only 5 classes with a larger number of images.

The third dataset, Dataset 3, was created by Kaggle user Piyush Kumar in 2019. It contains approximately 17,200 images of various animals, including butterfly, cats, cows, elephants, hens, horses, monkeys, pandas, sheep, spiders, squirrels, among others. Size of the images are ranging from 201 x 300 to 500 x 374 pixels.

## 3. Pros and Cons of existing solutions

### Pros
One of the biggest pros of existing manual animal classification is their ability to provide accurate results. Trained experts in animal classification can achieve high levels of accuracy, especially for challenging or novel species. Humans can adapt to new situations and recognize new animals without requiring extensive retraining.
### Cons
Manual classification is a timeconsuming process that requires significant human effort, limiting its scalability. Different individuals may classify
the same image differently, leading to potential inconsistencies and biases. 

## 4. CNN Models

ResNet18, ShuffleNetV2 and MobileNetV2 are three models which has been used for this study for performance comparision on different datasets. 

## 3.Methodology:

**Data Cleaning:**
	These datasets have some noise. We are going to clean the dataset and after that we will split these datasets into training and testing set. Training set will 	be only used for training the dataset and testing dataset will be used for the evaluation of the dataset.

**Data Processing:**
	Resize: consists of variable-resolution images, while our system requires a constant input dimensionality. There for we are going to down sampled the images to  	 fixed resolution which is suitable for architecture.
	Tensor: convert the NumPy images to torch images.
	Normalize: change the range of pixels.
	ColorJitter: To generate images with randomly brightness contrast and saturation 
	Scale Conversion: This is a trial. First, we train the model with Gray scale image if it is not giving good result we will move to RGB.
	Moreover, to handle imbalance dataset we will use other augmentation techniques if needed.
	
**Training models:**
We have chosen MobilenetV2, ResNet18 and DenseNet121. The reason for choosing this architecture is these are light weight architecture and computationally 	light except DenseNet121. These architectures are commonly used architecture for classification because of the accuracy. We will also perform the transfer 	   	learning for any 2 Architecture. We will use different hyperparameters such as Learning Rate, Cost functions, Batch Size, Number of iterations (Epochs) in 	   training, Optimizers.

**Model selection & Evaluation metrics:**
We will train the model till 100 epochs (may vary if taking time in training).Using tensor board loss and accuracy analysis, best epoch which has higher 	accuracy is going to selected for the predictions.
To compare two different model, we will use confusion metric, F1 score, Precision, recall and TSNE.

**Applications of Derived results:**
In term of research, these all results can be used for research purpose not only in animal classification but also for the other image classifications. For instance, Results of Accuracy and error rates for different architecture on different types dataset can be used for selection of architecture for future research. Moreover, this also shows what types of hyper parameter can give better results. In terms of real application, model can be used for the wildlife conservation, Zoo and agriculture. 

