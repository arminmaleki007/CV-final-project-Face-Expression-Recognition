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

I am following the work of https://github.com/SainingZhang/DDAMFN/tree/main. The architectural overview of the DDAMFN, illustrated in the next figure, consists of two primary components: the MFN and the DDAN. First, facial images are processed by the MFN, which generates initial feature maps. These feature maps are then refined through the DDAN, which creates attention maps in both vertical and horizontal directions. Finally, the attention maps are reshaped to the required dimensions, and a fully connected layer predicts the expression category of the images.

![image](https://github.com/user-attachments/assets/0d1ea4c4-3ab2-4a74-87c1-ee4c66ac581b)
### MFN: 
Considering the potential overfitting issues associated with the use of heavy network architectures on small FER datasets, a lightweight network, MobileFaceNet, was adopted as the foundation. As illustrated in the next Figure a combination of two primary building blocks, a residual bottleneck and a non-residual block, was employed.

![image](https://github.com/user-attachments/assets/e9c7bee1-e104-4175-8596-43c98aecc226)

The MFN architecture incorporates both residual and non-residual bottleneck blocks to balance feature extraction and information flow, enhancing its effectiveness for facial expression recognition (FER). Residual blocks mitigate degradation and improve gradient flow, while non-residual blocks enhance representational capacity to capture diverse facial features. The MixConv operation, with multiple kernel sizes illustrated in the next Figure, is integrated into the bottleneck, enabling the MFN to extract diverse and detailed features, surpassing MobileFaceNet. PreLU activation further improves facial feature extraction. Additionally, the network depth is optimized, and a coordinate attention mechanism is introduced to model long-range dependencies and provide accurate positional information, outperforming CBAM in FER tasks.

![image](https://github.com/user-attachments/assets/dc692715-3711-4f0a-9b62-34bf1557a054)
### DDAN:
The DDAN comprises multiple independent dual-direction attention (DDA) heads, each designed to capture long-range dependencies within the network. Based on the coordinate attention mechanism, DDA heads generate direction-aware feature maps from both horizontal and vertical directions. Instead of average pooling, linear GDConv is used to assign varying importance to different spatial positions, emphasizing key facial areas and enhancing the attention mechanism's discriminative power.

Each DDA head produces two attention maps, which are combined into a final attention map matching the input feature map's size. The most salient attention map is selected, and the final output is obtained through element-wise multiplication of the input feature map with the selected attention map. To ensure each DDA head focuses on distinct facial regions, a novel attention loss function is introduced, further optimizing the attention mechanism for improved feature extraction.
### Loss: 
Attention loss: The Mean Squared Error (MSE) loss is calculated between each pair of attention maps generated from different dual-direction heads. The attention loss is then defined as the reciprocal of the sum of these MSE losses, which can be mathematically expressed as follows:

$L_{att} = \frac{1}{\sum_{i=0}^n \sum_{k=0, i \neq k}^n \text{MSE}(a_i, a_k)} (i \neq j)$

where 𝑛 is the number of attention heads. $a_i$ and $a_k$ are attentions maps yielded from two different heads.

The feature map of size 7 × 7 × 512, obtained from the DDAN, undergoes a linear GDConv layer and a linear layer. This transformed feature map is then reshaped to a 512 d vector. The class confidence is obtained via a fully connected layer. Regarding the loss function, the standard cross-entropy loss is employed in the training process. This loss function effectively measures the discrepancy between predicted class probabilities and the ground truth labels, facilitating the optimization of the model’s parameters. The overall loss function can be expressed as follows:

$L = L_{cls} + \lambda_a L_{att}$

where $L_{cls}$ stands for standard cross entropy loss and $L_{att}$ is attention loss. $\lambda_a$ is a hyperparameter. The default of $\lambda_a$ is 0.1.
### Training
