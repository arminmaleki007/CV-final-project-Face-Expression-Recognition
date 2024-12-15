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

where 𝑛 is the number of attention heads. $a_i$ and $a_k$ are attention maps yielded from two different heads.

The feature map of size 7 × 7 × 512, obtained from the DDAN, undergoes a linear GDConv layer and a linear layer. This transformed feature map is then reshaped to a 512 d vector. The class confidence is obtained via a fully connected layer. Regarding the loss function, the standard cross-entropy loss is employed in the training process. This loss function effectively measures the discrepancy between predicted class probabilities and the ground truth labels, facilitating the optimization of the model’s parameters. The overall loss function can be expressed as follows:

$L = L_{cls} + \lambda_a L_{att}$

where $L_{cls}$ stands for standard cross entropy loss and $L_{att}$ is attention loss. $\lambda_a$ is a hyperparameter. The default of $\lambda_a$ is 0.1.

You can see the original picture (first column), the picture heatmap after the MFN backbone (second column), and the picture heatmap after the DDAN network (third column) in the next picture.

![combined_reversed_side_by_side](https://github.com/user-attachments/assets/05baa1db-928f-4cdd-8405-36d676b92fe4)

### Training
I trained the network with the dataset 8 times. I used my laptop with TRX 3060 with 6GB GPU. I used batch-size=64 because of the GPU limitations.

First of all, I used two heads for the attention mechanism. Then I used four heads. Then I changed the learning rate from 0.01 to 0.001. The next step I took was to use preprocessing data as Retinaface. In this method, the output is the picture with a boundary box for the face with 5 dots (two for the eyes, one for the nose, and two for the mouth). 

https://github.com/deepinsight/insightface/tree/master/detection/retinaface

This method requires a label for pictures to train, and it is not available for the Ferplus dataset. I just used their trained model to test the dataset to draw the boundary box with dots on the original pictures as you can see in the next figure. Unfortunately, the method could not detect faces for all of the pictures in the dataset.

![augmented_6](https://github.com/user-attachments/assets/f9e3b62b-caa4-401b-a720-7fbeb632a0b6)

I used data augmentation like random rotation, flip, and color jittering for training and validation. However, the training process overfitted in the early epochs. So, I trained without data augmentation for the original and processed datasets. At last. I used batch_size = 32 as the last step. You can see the training and validation plots for loss and accuracy, the validation confusion matrix, and the test confusion matrix for the best training for the original dataset in the following figures.

![output](https://github.com/user-attachments/assets/0673bc86-54b8-4df7-ace4-f7fe2921d078) ![output (1)](https://github.com/user-attachments/assets/f9cae5f7-ff19-40db-b1ef-3eb59dcc24b8) ![ferPlus_epoch3_acc0 771_bacc0 767](https://github.com/user-attachments/assets/3727d5e7-5632-4c30-8aa5-4eb1e1e90126) ![ferPlus_acc0 8167_bacc](https://github.com/user-attachments/assets/40a20f7c-b90f-401a-87c9-0535c575eeb4)

You can see the training and validation plots for loss and accuracy, the validation confusion matrix, and the test confusion matrix for the best training for the processed dataset t in the following figures.

![output](https://github.com/user-attachments/assets/6f086aab-2e00-4372-8840-4133499d25d8) ![output (1)](https://github.com/user-attachments/assets/2ee1fc54-f154-4404-b1a1-a9d61ff2a533) ![ferPlus_epoch5_acc0 7667_bacc0 7611](https://github.com/user-attachments/assets/f1558c3a-6efd-44ee-8b50-2c67c7697a54) ![ferPlus_acc0 8262_bacc](https://github.com/user-attachments/assets/fa2c48e5-1f5e-4425-949c-51b2618e499d)



As you can in the training plot, the network is overfitted. 

### Code instructions
To train, first you need to download the dataset, run ferPlus_train.py, and choose the dataset directory (default='/data/ferPlus/'), number of epochs (default =80), batch_size (defult=64), number of heads (defult=4), number of workers (defult=8), and Initial learning rate (defult=0.01). The training process will save the trained network, and validation confusion matrix if the validation accuracy is more than a threshold. 

To test, first, you need to extract the weights from the checkpoints folder, then you need to run ferPlus_test.py and choose the dataset directory (default='/data/ferPlus/'), model_path (default='./checkpoints/original/original.pth'), batch_size (defult=64), number of heads (defult=4), and number of workers (defult=8).

## Part 4: Second Update
I want to study the effect of dimension changing of the picture in the test results. To do so, I created a new Python code called **ferPlus_test_resiez.py**. This code has added parts to the ferPlus_test. The testing image first will be downsized using the cv2.resize function. You can choose what size the image changes to by defining the resize argument. The default value is 112 which is the original dataset dimension. Then, the downsized image will be upsized to a 112*112 image. I used a different cv2.resize options with different downsized image dimensions. You can see the results as follows.

|Method\resize value|  112 |  96  |  84  |  64  |  56  |  48  |  32  |  28  |  16  |  14  |
|-------------------|------|------|------|------|------|------|------|------|------|------|
|    Both Cubic     |81.08%|81.05%|81.00%|81.02%|81.36%|81.08%|79.32%|77.36%|57.65%|52.25%|
|    Both Linear    |81.08%|81.39%|81.30%|81.25%|81.21%|81.00%|79.12%|77.55%|57.93%|53.59%|
|    Both Lanczos4  |81.08%|81.16%|81.05%|80.91%|81.11%|80.83%|79.18%|77.41%|53.82%|49.45%|
|    Cubic Linear   |81.08%|81.19%|81.19%|81.47%|81.58%|81.30%|78.67%|77.55%|57.93%|52.20%|

As you can see, all the methods performed similarly. All the methods did the same testing value as the original image dimensions until using 32 * 32 resizing, and smaller values performed worse. Therefore we can state that 48 * 48 is the threshold for resizing in testing. The Lanczos4 method performed the worst. You can see the results in "checkpoints/test results".

Another task that I have done is to test the model with a laptop webcam. You can run the test by running the **ferPlus_test_camera.py** code. 
First I start the webcam. Then I added a 224 * 224 bounding box overlay on the camera feed and allowed capturing the face within the box, Then I resized the image inside the boundary box to 112 * 112. Next, I converted the image to grayscale. At last, I loaded the model to test the image. The predicted classes are shown in the top left of the camera feed. 

Unfortunately, the testing accuracy in the webcam feed seems low. This is the result of overfitting. 
## Part 5: The final update

For the last update, I studied the test accuracy and confusion matrix for two different subsets from the original test dataset. I divided the test dataset into **Kids** images and **adults** images. The main purpose behind this study is that adults have more developed muscles in their faces, they learned how to express their feelings over time, and their face posture has more features to detect feelings. Therefore, detecting emotional features in adults' faces should be easier than kids', and since my dataset contains both group images, I want to study the effect of each group on the testing accuracy. 

The first objective was to study the accuracy of the data based on age groups. However, I decided to study the accuracy just for kids and adults because of the lack of data in certain groups, and lack of labels to determine the age group.

Here is a brokedown number of pictures in each sub-dataset:

|Age group\ number of data| Total | Angry | Contempt | Disgust | Fear | Happy | Neutral | Sad | Surprise |
|-------------------------|-------|-------|----------|---------|------|-------|---------|-----|----------|
|           Kids          | 620   | 51    | 4        | 6       | 7    | 170   | 153     | 129 | 100      |
|          Adults         | 2947  | 267   | 25       | 14      | 91   | 759   | 1121    | 320 | 350      |

As you can see, the amount of data for kids is less than that for adults. This definitely has a result on the accuracy, because one wrong sample decreases the accuracy of more than adults. You can see the accuracy and confusion matrix. The first image is for adults and the image is for kids.

![ferPlus_acc0 8276_bacc](https://github.com/user-attachments/assets/a96d9eb2-8521-4e67-b538-55440c8fff76)

![ferPlus_acc0 8048_bacc](https://github.com/user-attachments/assets/5297ca6c-4a9c-4011-8f85-92894a8b86a1)

As you can see, the accuracy for adults is 2% more than the accuracy for kids. In addition, all the classes have more accuracy for adults than the kids except **Angry** class. This is because almost all of the kids in the angry class are crying, and it is easier to detect. However, less data in the kids' dataset makes it harder to get a reasonable conclusion.

In summary, for this project, I used a vision transformer-based model to classify facial expressions. Even though the model was overfitted in early epochs, the results were satisfying. In addition, I studied the effect of resizing images and the age group on the accuracy. At last, I developed a system to detect facial expressions in real-time with the camera. 
