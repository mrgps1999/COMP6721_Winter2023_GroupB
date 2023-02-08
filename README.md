# Animal Classification Application

## 1.	Problem Statement and Introduction:

Research indicates that our species are facing a biodiversity crisis and a human-caused global mass extinction event. Threats to endangered wildlife species must be eradicated in order to preserve biodiversity. The fragmentation and destruction of habitat, the overuse of resources and human-caused climate change are some of the main causes of extinction and biodiversity loss. Popular wildlife conservation programs use camera-trap studies, in which motion-activated cameras are set up at study sites to record images of moving animals. The resulting images are then applied in a variety of ways, including tracking specific animals, estimating species richness and abundance, and maintaining an eye for invasive species. These conventional methods of animal observation, such as track censuses or direct counts, all of these applications are intrusive, expensive, and takes time. Animal classification application refer to software tools or systems used to identify and categorize different species of animals. We proposed Animal classification using convolution neural network and for that we are going to use 3 Datasets with different classes and 3 CNN architectures like ResNet18, MobilenetV2, DenseNet121. Moreover, we will use two models for transfer learning. To, evaluate all this models, we are going to use different evaluation metric. Main challenge in animal classification is unbalanced dataset. Secondly, Quality of images is second major issue. To elaborate, some dataset has images which are less than 200X200. These applications can be used for scientific research, wildlife management, and monitoring. 

## 2.	Image data selection 

There are many datasets available for the animal classification datasets among that we selected below 2 datasets with different classes and different number of images per class.

|Name 		|Number of Images|	 Classes|   format|    link| 
|Animal		|			15.1k	|25		|jpg|		|
|Animal Image dataset|			17.2k|	12 |  		jpg| |
|Kaggle cats and dogs|			24.9k|	2 |    	         jpg| |

NAME AND THEIR CLASSES


## 3.	 Methodology:

_ **Data Cleaning:**

These datasets have some noise. We are going to clean the dataset and after that we will split these datasets into training and testing set. Training set will be only used for training the dataset and testing dataset will be used for the evaluation of the dataset.
_ **Data Processing:**
Resize: consists of variable-resolution images, while our system requires a constant input dimensionality. There for we are going to down sampled the images to fixed resolution which is suitable for architecture.
Tensor: convert the NumPy images to torch images.
Normalize: change the range of pixels.
ColorJitter: To generate images with randomly brightness contrast and saturation 
Scale Conversion: This is a trial. First, we train the model with Gray scale image if it is not giving good result we will move to RGB.
Moreover, to handle imbalance dataset we will use other augmentation techniques if needed.
_ **Training models:**
		We have chosen MobilenetV2, ResNet18 and DenseNet121. The reason for choosing this architecture is these are light weight architecture and computationally light except DenseNet121. These architectures are commonly used architecture for classification because of the accuracy. We will also perform the transfer learning for any 2 Architecture. We will use different hyperparameters such as Learning Rate, Cost functions, Batch Size, Number of iterations (Epochs) in training, Optimizers.

_ **Model selection & Evaluation metrics:**

We will train the model till 100 epochs (may vary if taking time in training). Using tensor board loss and accuracy analysis, best epoch which has higher accuracy is going to selected for the predictions.
To compare two different model, we will use confusion metric, F1 score, Precision, recall and TSNE.

_ **Applications of Derived results:**
In term of research, these all results can be used for research purpose not only in animal classification but also for the other image classifications. For instance, Results of Accuracy and error rates for different architecture on different types dataset can be used for selection of architecture for future research. Moreover, this also shows what types of hyper parameter can give better results. In terms of real application, model can be used for the wildlife conservation, Zoo and agriculture. 

