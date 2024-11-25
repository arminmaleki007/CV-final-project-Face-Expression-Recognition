# Facial Expression-Recognition
## Part 1: Conceptual design
The objective of this project is online facial expression recognition using the camera. We can use any camera-based system like a laptop webcam or a simple Raspberry Pi system to recognize facial expressions. The importance of this work relies on the fact that emotions change the facial expression of the face. Using this system can help us to know people's reactions in different situations.

To do this task, we need to design a computer vision method using deep learning to extract features from faces that are different for different facial expressions. For example. the shape of lips, eyes, brows, and face lines are different for different facial expressions. The color of features mustn't affect the classification tasks, and it is better to use gray-scale pictures to speed up the training process. Detecting these changes is necessary to do the task. At last, we need to use these features to **classify** the facial expression of the person in the camera view. 

I am thinking about two solutions for doing this task. The first solution is using deep CNN networks for feature extraction and MLP networks for classification. In addition, we need to define a class for each expression in the dataset plus a not detected class for expressions with low similarities to the goal classes. We need to tune the system's hyperparameters (like the number of layers, the number of nodes in each layer, batch size, activation function, and even the optimizer) for the best performance. The second solution is to use an encoder-decoder algorithm using transformers. This architecture uses a latent space for each class of facial expressions. Each point in the latent space corresponds to a class. This is a proposal for the solution method and requires a deeper understanding of the methods and implementation. 

I found a dataset with 8 different classes for facial expressions. The dataset contains Angry, Contempt, Disgust, Fear, Happy, Neutral, Sadness, and Surprise. There are multiple pictures from video frames of faces from different people with different facial expressions. We need to divide the dataset by people, not pictures into training (70%), validation (20%), and test sets (10%). This is because using the same person in training, validation, and testing, will result in overfitting without knowing. That being said, I believe we can find other datasets with more classes and more pictures by spending more time. On the other hand, we can use video frames as input for RNN networks after feature extraction using CNN. This will help the classification problem by using data series with RNN networks. In addition, I plan to use data augmentation. This increases data in the dataset, which will improve the performance of the system by allowing us to use deeper networks without overfitting.

For this problem, we need to use cross-entropy loss. In addition, we need to use accuracy and loss as a metric for the performance evaluation. We can use the confusion matrix as a good metric too. We can try different activation functions for network layers. However, I think ReLU or Leaky ReLU would work fine for all layers except the last layer which we need to use the sigmoid function. In addition, ADAM optimizer may work best for this task. 

My plan is to use my laptop for training. My laptop has a 6 GB Nvidia RTX 3060 GPU which I think is enough for training for this task. The alternative environment is CRC. In addition, I think Google Colab may work too. However, it will be slower than other environments.


## Part 2: Data acquisition and preparation
There are a number of datasets available on the internet. However, the public datasets are limited. After lots of research, I want to use the following dataset for this project:
https://www.kaggle.com/datasets/subhaditya/fer2013plus
This dataset contains 8 classes. The number of images for each class is much higher than CK+. The classes are included as Angry, Contempt, Disgust, Fear, Happy, Neutral, Sad, and Surprise. 

The dataset consists of 66387 images for training, 8341 images for validation, and 3586 images for testing. Therefore, there are a total of 78314 images which is divided into 84% for training, 11% for validation, and 5% for the test. You can see the class details of the dataset in the following table.  

| Class/Data | Total | Angry |contempt| Disgust | Fear | Happy | Neutral |  Sad  | Surprise |
|------------|-------|-------|--------|---------|------|-------|---------|-------|----------|
|  Training  | 66387 | 8000  |  8000  |  8000   | 8000 |  8000 |  10379  | 8000  |   8000   |
| Validation | 8341  | 1000  |  1000  |  1000   | 1000 |  1000 |  1341   | 1000  |   1000   |
|    Test    | 3586  |  332  |   30   |   21    |  98  |  929  |  1274   |  449  |    550   |
|   Total    | 78314 | 9332  |  9032  |   9021  | 9098 |  9929 |  12994  | 9449  |   9550   |

Each image in the dataset is a 112*112 pixels grayscale image of a person's face. Images are collected from the internet and have been classified for the dataset by hand. In addition to the dataset, I will use data augmentation like flip and rotation to increase the number of data samples for a better solution. 
## Part 3: First update

For the training, I am using a CNN-based transformer architecture. Since the success of attention mechanisms in large language models, this mechanism has gotten lots of attention in other fields. Researchers have explored the integration of attention mechanisms into deep Convolutional Neural Networks (CNNs) to extract more informative features from images. Transformers have recently emerged as a dominant framework across various tasks, outperforming traditional models like Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs). Leveraging multi-head attention mechanisms, Transformers have demonstrated exceptional performance across multiple domains. 

I am following the work of https://github.com/SainingZhang/DDAMFN/tree/main.
